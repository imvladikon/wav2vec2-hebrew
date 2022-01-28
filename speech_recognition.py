#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import Optional, Union

from datasets import load_metric
import torch
from pydub.silence import split_on_silence
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    AutomaticSpeechRecognitionPipeline,
    AutoTokenizer,
)
from pydub import AudioSegment


class HebrewSpeechRecognitionPipeline(AutomaticSpeechRecognitionPipeline):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[
                Union[str, os.PathLike]] = "imvladikon/wav2vec2-large-xlsr-53-hebrew",
            *args,
            **kwargs,
    ):

        super().__init__(
            feature_extractor=AutoFeatureExtractor.from_pretrained(
                pretrained_model_name_or_path
            ),
            model=Wav2Vec2ForCTC.from_pretrained(
                pretrained_model_name_or_path
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path
            ),
            *args,
            **kwargs,
        )
        self.model_name = pretrained_model_name_or_path
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.metric_fn = None
        self.postprocess_text_fn = lambda x: x.replace("[PAD]", "")

    @torch.inference_mode()
    def __call__(self, file, labels=None, *args, **kwargs):
        is_one_sample = not isinstance(file, str)
        results = super(HebrewSpeechRecognitionPipeline, self).__call__(
            file, *args, **kwargs
        )
        if not is_one_sample:
            results = [results]
            if labels is not None:
                labels = [labels]
        if self.metric_fn is None and labels is not None:
            self.metric_fn = load_metric("wer")
        for idx, result in enumerate(results):
            result["text"] = self.postprocess_text_fn(result["text"])
            if labels is not None:
                label = labels[idx]
                result["wer"] = self.metric_fn.compute(
                    predictions=[result["text"]],
                    references=[label],
                )
        return results


if __name__ == '__main__':
    # asr = HebrewSpeechRecognitionPipeline()
    # filename = "./samples/013b882862704b7792ce16cd944b98470.wav"
    # output = asr(filename, labels="שלום תלמידים שמי עופרה והיום אנחנו נדבר על")
    # print(output)
    #
    # asr = HebrewSpeechRecognitionPipeline()
    filename = "./1.m4a"
    AudioSegment.from_file(filename).export("./samples/1.wav", format="wav")
    filename = "./samples/1.wav"
    # output = asr(filename)
    # print(output)
    chunks = split_on_silence(
        AudioSegment.from_file(filename),
        # must be silent for at least half a second
        min_silence_len=100,

        # consider it silent if quieter than -16 dBFS
        # silence_thresh=-16
    )
    for i, chunk in enumerate(chunks):
        chunk.export(f"{i}.wav", format="wav")
