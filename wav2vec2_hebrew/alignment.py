import os
from dataclasses import dataclass

import torch
import torchaudio
from tqdm import tqdm

from transformers import AutoConfig, AutoModelForCTC, AutoProcessor


class Wav2Vec2Aligner:
    def __init__(self,
                 model_name: str,
                 input_sample_rate: int,
                 use_cuda: bool,
                 sampling_rate: int = 16_000):
        self.use_cuda = use_cuda
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.eval()
        if self.use_cuda:
            self.model.to(device="cuda")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.resampler = torchaudio.transforms.Resample(input_sample_rate, 16_000)
        blank_id = 0
        vocab = list(self.processor.tokenizer.get_vocab().keys())
        for i in range(len(vocab)):
            if vocab[i] == "[PAD]" or vocab[i] == "<pad>":
                blank_id = i
        print("Blank Token id [PAD]/<pad>", blank_id)
        self.blank_id = blank_id
        self.sampling_rate = sampling_rate

    def speech_file_to_array_fn(self, wav_path):
        speech_array, sampling_rate = torchaudio.load(wav_path)
        speech = self.resampler(speech_array).squeeze().numpy()
        return speech

    @torch.inference_mode()
    def align_single_sample(self, item):
        blank_id = self.blank_id
        transcript = "|".join(item["sentence"].split(" "))
        if not os.path.isfile(item["wav_path"]):
            print(item["wav_path"], "not found in wavs directory")

        speech_array = self.speech_file_to_array_fn(item["wav_path"])
        inputs = self.processor(speech_array,
                                sampling_rate=self.sampling_rate,
                                return_tensors="pt",
                                padding=True)
        data_size = inputs.input_values.size(1)
        if self.use_cuda:
            inputs = inputs.to(device="cuda")
        logits = self.model(inputs.input_values).logits

        # get the emission probability at frame level
        emissions = torch.log_softmax(logits, dim=-1)
        emission = emissions[0].cpu().detach()

        # get labels from vocab
        labels = ([""] + list(self.processor.tokenizer.get_vocab().keys()))[
                 :-1
                 ]  # logits don't align with the tokenizer's vocab

        dictionary = {c: i for i, c in enumerate(labels)}
        tokens = []
        for c in transcript:
            if c in dictionary:
                tokens.append(dictionary[c])

        def get_trellis(emission, tokens, blank_id=0):
            """
            Build a trellis matrix of shape (num_frames + 1, num_tokens + 1)
            that represents the probabilities of each source token being at a certain time step
            """
            num_frames = emission.size(0)
            num_tokens = len(tokens)

            # Trellis has extra diemsions for both time axis and tokens.
            # The extra dim for tokens represents <SoS> (start-of-sentence)
            # The extra dim for time axis is for simplification of the code.
            trellis = torch.full((num_frames + 1, num_tokens + 1), -float("inf"))
            trellis[:, 0] = 0
            for t in range(num_frames):
                trellis[t + 1, 1:] = torch.maximum(
                    # Score for staying at the same token
                    trellis[t, 1:] + emission[t, blank_id],
                    # Score for changing to the next token
                    trellis[t, :-1] + emission[t, tokens],
                )
            return trellis

        trellis = get_trellis(emission, tokens, blank_id)
        trellis_size = trellis.size(0)

        @dataclass
        class Point:
            token_index: int
            time_index: int
            score: float

        def backtrack(trellis, emission, tokens, blank_id=0):
            """
            Walk backwards from the last (sentence_token, time_step) pair to build the optimal sequence alignment path
            """
            # Note:
            # j and t are indices for trellis, which has extra dimensions
            # for time and tokens at the beginning.
            # When referring to time frame index `T` in trellis,
            # the corresponding index in emission is `T-1`.
            # Similarly, when referring to token index `J` in trellis,
            # the corresponding index in transcript is `J-1`.
            j = trellis.size(1) - 1
            t_start = torch.argmax(trellis[:, j]).item()

            path = []
            for t in range(t_start, 0, -1):
                # 1. Figure out if the current position was stay or change
                # Note (again):
                # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
                # Score for token staying the same from time frame J-1 to T.
                stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
                # Score for token changing from C-1 at T-1 to J at T.
                changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

                # 2. Store the path with frame-wise probability.
                prob = emission[
                    t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
                # Return token index and time index in non-trellis coordinate.
                path.append(Point(j - 1, t - 1, prob))

                # 3. Update the token
                if changed > stayed:
                    j -= 1
                    if j == 0:
                        break
            else:
                raise ValueError("Failed to align")
            return path[::-1]

        path = backtrack(trellis, emission, tokens, blank_id)

        @dataclass
        class Segment:
            label: str
            start: int
            end: int
            score: float

            def __repr__(self):
                return f"{self.label}\t{self.score:4.2f}\t{self.start * 20:5d}\t{self.end * 20:5d}"

            @property
            def length(self):
                return self.end - self.start

        def merge_repeats(path):
            """
            Merge repeated tokens into a single segment. Note: this shouldn't affect repeated characters from the
            original sentences (e.g. `ll` in `hello`)
            """
            i1, i2 = 0, 0
            segments = []
            while i1 < len(path):
                while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                    i2 += 1
                score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
                segments.append(
                    Segment(
                        transcript[path[i1].token_index],
                        path[i1].time_index,
                        path[i2 - 1].time_index + 1,
                        score
                    )
                )
                i1 = i2
            return segments

        segments = merge_repeats(path)

        def merge_words(segments, ratio, separator="|"):
            words = []
            i1, i2 = 0, 0
            while i1 < len(segments):
                if i2 >= len(segments) or segments[i2].label == separator:
                    if i1 != i2:
                        segs = segments[i1:i2]
                        word = "".join([seg.label for seg in segs])
                        score = sum(seg.score * seg.length for seg in segs) / sum(
                            seg.length for seg in segs)
                        words.append(Segment(word, ratio * segments[i1].start,
                                             ratio * segments[i2 - 1].end, score))
                    i1 = i2 + 1
                    i2 = i1
                else:
                    i2 += 1
            return words

        segments = merge_words(segments, ratio=data_size / (trellis_size - 1))

        return segments

    def show_segments(self, audio, segments):
        import IPython
        from IPython.display import display

        for segment in segments:
            print(segment.label)
            segment_array = audio[:, int(segment.start):int(segment.end)].numpy()
            display(IPython.display.Audio(segment_array, rate=self.sampling_rate))

    def align_data(self, wav_file, text):
        tasks = []
        for sentence in text.split("\n"):
            sentence = sentence.strip()
            tasks.append({"sentence": sentence, "wav_path": wav_file})
        results = []
        for task in tqdm(tasks):
            segments = self.align_single_sample(task)
            results.append({
                "sentence": task["sentence"],
                "segments": segments
            })
        return results


class HebrewWav2Vec2Aligner(Wav2Vec2Aligner):

    def __init__(self,
                 input_sample_rate: int,
                 use_cuda: bool,
                 model_name: str = "imvladikon/wav2vec2-xls-r-300m-hebrew"):
        super().__init__(model_name=model_name,
                         input_sample_rate=input_sample_rate,
                         use_cuda=use_cuda)
