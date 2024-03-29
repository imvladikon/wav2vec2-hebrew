# !/usr/bin/env python
# coding=utf-8
import functools
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio
from datasets import DatasetDict, load_dataset, load_metric
import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback, TrainingArguments,
    Wav2Vec2Processor,
    set_seed,
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import AutoProcessor

try:
    import bitsandbytes as bnb

    BNB_AVAILABLE = True
except:
    BNB_AVAILABLE = False
    logger.warning(
        "bitsandbytes is not installed, you will not be able to use it for training optimization.")
try:
    import wandb

    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False
    logger.warning(
        "wandb is not installed, you will not be able to log training progress.")

try:
    from torch_audiomentations import (
        Compose,
        AddGaussianNoise,
        AddGaussianSNR,
        ClippingDistortion,
        FrequencyMask,
        Gain,
        LoudnessNormalization,
        Normalize,
        PitchShift,
        PolarityInversion,
        Shift,
        TimeMask,
        TimeStretch,
    )

    AUDIOMENTATIONS_AVAILABLE = True
except:
    AUDIOMENTATIONS_AVAILABLE = False
    logger.warning(
        "torch_audiomentations is not installed, you will not be able to use it for audio augmentations.")


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout ratio for the attention probabilities."},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout ratio for activations inside the fully connected layer."
        },
    )
    feat_proj_dropout: float = field(
        default=0.0, metadata={"help": "The dropout ratio for the projected features."}
    )
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": "Probability of each feature vector along the time axis to be chosen as the start of the vector"
                    "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature"
                    "vectors will be masked along the time axis."
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={
            "help": "Probability of each feature vector along the feature axis to be chosen as the start of the vector"
                    "span to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature bins will be masked along the time axis."
        },
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    ctc_loss_reduction: Optional[str] = field(
        default="mean",
        metadata={
            "help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_path: str = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        }
    )
    dataset_name: str = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    dataset_config_name: str = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    text_column_name: str = field(
        default="text",
        metadata={
            "help": "The name of the dataset column containing the text data. Defaults to 'text'"
        },
    )
    wav_filesize_column_name: str = field(
        default=None,
        metadata={
            "help": "The name of the dataset column containing the wav filesize. Defaults is None"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    chars_to_ignore: Optional[List[str]] = list_field(
        default=None,
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    eval_metrics: List[str] = list_field(
        default=["wer"],
        metadata={
            "help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"
        },
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": "Filter audio files that are longer than `max_duration_in_seconds` seconds to 'max_duration_in_seconds`"
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={
            "help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"
        },
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to only do data preprocessing and skip training. "
                    "This is especially useful when data preprocessing errors out in distributed training due to timeout. "
                    "In this case, one should run the preprocessing in a non-distributed setup with `preprocessing_only=True` "
                    "so that the cached datasets can consequently be loaded in distributed training"
        },
    )
    print_samples: bool = field(
        default=False,
        metadata={
            "help": "Print row with validation inference results to stdout after each epoch"
        },
    )
    use_augmentations: bool = field(
        default=False,
        metadata={
            "help": "Use data augmentation during training"
        },
    )
    use_auth_token: str = field(
        default="",
        metadata={
            "help": "If :obj:`True`, will use the token generated when running"
                    ":obj:`transformers-cli login` as HTTP bearer authorization for remote files."
        },
    )
    unk_token: str = field(
        default="[UNK]",
        metadata={"help": "The unk token for the tokenizer"},
    )
    pad_token: str = field(
        default="[PAD]",
        metadata={"help": "The padding token for the tokenizer"},
    )
    word_delimiter_token: str = field(
        default="|",
        metadata={"help": "The word delimiter token for the tokenizer"},
    )
    phoneme_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "The target language that should be used be"
                    " passed to the tokenizer for tokenization. Note that"
                    " this is only relevant if the model classifies the"
                    " input audio to a sequence of phoneme sequences."
        },
    )


class Augmentator:

    def __init__(
            self,
            apply_gaussian_noise_with_p=0.1,
            apply_gain_with_p=0.1,
            apply_pitch_shift_with_p=0.1,
            apply_time_stretch_with_p=0.1,
            augment_proba=0.1,
            sample_rate=16_000
    ):
        self.augmentator_fn = None
        self.sample_rate = sample_rate
        self.augment_proba = augment_proba
        all_p = (
                apply_gaussian_noise_with_p
                + apply_gain_with_p
                + apply_pitch_shift_with_p
                + apply_time_stretch_with_p
        )
        if AUDIOMENTATIONS_AVAILABLE and all_p > 0:
            self.augmentator_fn = Compose([
                TimeStretch(min_rate=0.8, max_rate=1.2, leave_length_unchanged=False,
                            p=apply_time_stretch_with_p),
                PitchShift(min_semitones=-1, max_semitones=1,
                           p=apply_pitch_shift_with_p),
                Gain(min_gain_in_db=-1, max_gain_in_db=1, p=apply_gain_with_p),
                AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.001,
                                 p=apply_gaussian_noise_with_p),
            ])

    def __call__(self, input_values: List[float], *args, **kwargs):
        if AUDIOMENTATIONS_AVAILABLE and self.augmentator_fn is not None:
            return self.augmentator_fn(samples=np.array(input_values),
                                       sample_rate=self.sample_rate).tolist()
        else:
            return input_values


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: 'AutoProcessor'
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    augmentator_fn: Optional[Callable] = None
    use_augmentations: bool = False

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [
            {
                "input_values": self.augmentator_fn(feature["input_values"])
                if self.use_augmentations
                else feature["input_values"]}
            for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


def create_vocabulary_from_data(
        datasets: DatasetDict,
        text_column_name: str,
        train_split_name: str,
        word_delimiter_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        pad_token: Optional[str] = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(batch):
        all_text = " ".join(batch[text_column_name])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    print("extract chars")
    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets[train_split_name].column_names,
    )

    # take union of all unique characters in each dataset
    print("make vocab_set")
    vocab_set = functools.reduce(
        lambda vocab_1, vocab_2: set(vocab_1["vocab"][0]) | set(vocab_2["vocab"][0]),
        vocabs.values(),
    )

    vocab_dict = {v: k for k, v in enumerate(sorted(list(vocab_set)))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict


def speech_file_to_array_fn(batch, audio_column_name, dataset_path=""):
    if dataset_path and os.path.exists(dataset_path):
        dataset_path = os.path.join(dataset_path, batch[audio_column_name])
    elif isinstance(batch[audio_column_name], str):
        dataset_path = batch[audio_column_name]
    else:
        dataset_path = batch[audio_column_name]["path"]
        # https://huggingface.co/datasets/google/fleurs has a bit of a weird path
        if not Path(dataset_path).exists() and "path" in batch:
            dataset_path = str(Path(batch["path"]).parent / dataset_path)
    speech_array, sampling_rate = torchaudio.load(dataset_path)
    batch[audio_column_name] = {
        "array": speech_array[0].numpy(),
        "sampling_rate": sampling_rate,
    }
    return batch


class PrintSamplesPredictionCallback(TrainerCallback):

    def __init__(self, processor, eval_dataset):
        super(PrintSamplesPredictionCallback, self).__init__()
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.metric_fn = load_metric("wer")

    def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            model: Any,
            logs: Optional[Any] = None,
            **kwargs
    ):
        """
        :param args:
        :param state:
        :param control:
        :param model:
        :param logs:
        :param kwargs: 'tokenizer', 'optimizer', 'lr_scheduler', 'train_dataloader', 'eval_dataloader'
        :return:
        """
        if state.is_local_process_zero:
            columns = ["id", "prediction", "reference", "audio", "wer"]
            data = []
            for idx, row in enumerate(self.eval_dataset):
                input_dict = self.processor(row["input_values"],
                                            return_tensors="pt", padding=True)
                logits = model(input_dict.input_values.to(model.device)).logits
                pred_ids = torch.argmax(logits, dim=-1)[0]
                prediction = self.processor.decode(pred_ids)
                print(f"Prediction: {prediction}")
                reference = row['references'].lower()
                print(f"\nReference: {reference}")

                if WANDB_AVAILABLE:
                    audio = np.squeeze(row["audio"]["array"])
                    sample_rate = 16000
                    for sr_col in ["sampling_rate", "sample_rate", "rate"]:
                        if sr_col in row["audio"]:
                            sample_rate = row["audio"][sr_col]
                            break
                    audio = wandb.Audio(audio, sample_rate=sample_rate)
                    wer = self.metric_fn.compute(
                        predictions=[prediction],
                        references=[reference],
                    )
                    data.append([idx, prediction, reference, audio, wer])
            if WANDB_AVAILABLE:
                table = wandb.Table(data=data, columns=columns)
                wandb.run.log({"audio_predictions": table})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
            os.path.isdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    train_split_name = data_args.train_split_name
    eval_split_name = data_args.eval_split_name

    # 1. First, let's load the dataset
    raw_datasets = DatasetDict({
        train_split_name: None,
        eval_split_name: None,
    })

    if data_args.dataset_path:
        raw_datasets = load_dataset(
            "csv",
            data_files={
                train_split_name: os.path.join(data_args.dataset_path, "train-all.csv"),
                eval_split_name: os.path.join(data_args.dataset_path, "eval-all.csv"),
            },
        )

    if training_args.do_train:
        if raw_datasets[train_split_name] is None:
            raw_datasets[train_split_name] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=data_args.train_split_name,
                use_auth_token=data_args.use_auth_token,
            )

        if data_args.audio_column_name not in raw_datasets[train_split_name].column_names:
            raise ValueError(
                f"--audio_column_name '{data_args.audio_column_name}' not found in dataset. "
                "Make sure to set `--audio_column_name` to the correct audio column - one of "
                f"{', '.join(raw_datasets['train'].column_names)}."
            )

        if data_args.text_column_name not in raw_datasets[train_split_name].column_names:
            raise ValueError(
                f"--text_column_name {data_args.text_column_name} not found in dataset. "
                "Make sure to set `--text_column_name` to the correct text column - one of "
                f"{', '.join(raw_datasets['train'].column_names)}."
            )

        if data_args.max_train_samples is not None:
            raw_datasets[train_split_name] = raw_datasets[train_split_name].select(
                range(data_args.max_train_samples)
            )

        if data_args.wav_filesize_column_name is not None:
            raw_datasets[train_split_name] = raw_datasets[train_split_name].sort(
                data_args.wav_filesize_column_name, reverse=True)

    if training_args.do_eval:
        if raw_datasets[eval_split_name] is None:
            raw_datasets[eval_split_name] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=data_args.eval_split_name,
                use_auth_token=data_args.use_auth_token,
            )

        if data_args.max_eval_samples is not None:
            raw_datasets[eval_split_name] = raw_datasets[eval_split_name].select(
                range(data_args.max_eval_samples)
            )
        if data_args.wav_filesize_column_name is not None:
            raw_datasets[eval_split_name] = raw_datasets[eval_split_name].sort(
                data_args.wav_filesize_column_name, reverse=True)

    # save special tokens for tokenizer
    word_delimiter_token = data_args.word_delimiter_token
    unk_token = data_args.unk_token
    pad_token = data_args.pad_token

    # 3. Next, let's load the config as we might need it to create
    # the tokenizer
    # load config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_auth_token=data_args.use_auth_token,
    )

    # 4. Next, if no tokenizer file is defined,
    # we create the vocabulary of the model by extracting all unique characters from
    # the training and evaluation datasets
    # We need to make sure that only first rank saves vocabulary
    # make sure all processes wait until vocab is created
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    tokenizer_kwargs = {}

    # 5. Now we can instantiate the feature extractor, tokenizer and model
    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.
    with open(os.path.join(tokenizer_name_or_path, "vocab.json"), "r") as fin:
        print("loading tokenizer")
        print(fin.read())

    # load feature_extractor and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_auth_token=data_args.use_auth_token,
        **tokenizer_kwargs,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_auth_token=data_args.use_auth_token,
    )

    # adapt config
    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": model_args.activation_dropout,
        }
    )

    # create model
    model = AutoModelForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        use_auth_token=data_args.use_auth_token,
    )

    # freeze encoder
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # 6. Now we preprocess the datasets including loading the audio, resampling and normalization
    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`

    # make sure that dataset decodes audio with correct sampling rate

    # derive max & min input length for sample rate & max duration
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers

    # `phoneme_language` is only relevant if the model is fine-tuned on phoneme classification
    phoneme_language = data_args.phoneme_language

    if training_args.do_train:
        raw_datasets[train_split_name] = raw_datasets[train_split_name].map(
            speech_file_to_array_fn,
            num_proc=num_workers,
            fn_kwargs={"dataset_path": data_args.dataset_path,
                       "audio_column_name": audio_column_name},
        )
    if training_args.do_eval:
        raw_datasets[eval_split_name] = raw_datasets[eval_split_name].map(
            speech_file_to_array_fn,
            num_proc=num_workers,
            fn_kwargs={"dataset_path": data_args.dataset_path,
                       "audio_column_name": audio_column_name},
        )

    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio
        sample = batch[audio_column_name]

        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])

        # encode targets
        additional_kwargs = {}
        if phoneme_language is not None:
            additional_kwargs["phonemizer_lang"] = phoneme_language

        batch["labels"] = tokenizer(batch[data_args.text_column_name],
                                    **additional_kwargs).input_ids
        return batch

    print(f"Vectorizing")

    # TODO: workaround, sometimes happens for different options of the --do_train and --do_eval flags
    if "train" in raw_datasets and raw_datasets["train"] is None:
        raw_datasets.pop("train")
    if "validation" in raw_datasets and raw_datasets["validation"] is None:
        raw_datasets.pop("validation")

    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=num_workers,
            desc="preprocess datasets",
        )

    # 7. Next, we can prepare the training.
    # Let's use word error rate (WER) as our evaluation metric,
    # instantiate a data collator and the trainer

    # Define evaluation metrics during training, *i.e.* word error rate, character error rate
    eval_metrics = {metric: load_metric(metric) for metric in data_args.eval_metrics}

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        logger.info(
            f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}"
        )
        return

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        metrics = {
            k: v.compute(predictions=pred_str, references=label_str)
            for k, v in eval_metrics.items()
        }

        return metrics

    # Now save everything to be able to create a single processor later
    if is_main_process(training_args.local_rank):
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)

    try:
        processor = AutoProcessor.from_pretrained(training_args.output_dir)
    except (OSError, KeyError):
        warnings.warn(
            "Loading a processor from a feature extractor config that does not"
            " include a `processor_class` attribute is deprecated and will be removed in v5. Please add the following "
            " attribute to your `preprocessor_config.json` file to suppress this warning: "
            " `'processor_class': 'Wav2Vec2Processor'`",
            FutureWarning,
        )
        processor = Wav2Vec2Processor.from_pretrained(training_args.output_dir)

    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor,
        augmentator_fn=Augmentator(),
        use_augmentations=data_args.use_augmentations
    )

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if n not in decay_parameters
            ],
            "weight_decay": 0.0,
        },
    ]
    trainer_kwargs = {}
    if BNB_AVAILABLE:
        optimizer = bnb.optim.Adam8bit(
            params=optimizer_grouped_parameters,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
        )
        trainer_kwargs["optimizers"] = (optimizer, None)

    samples_to_log = [
        {
            **vectorized_datasets[eval_split_name][i],
            "references": raw_datasets[eval_split_name][i][data_args.text_column_name],
            "audio": raw_datasets[eval_split_name][i][data_args.audio_column_name],
        } for i in range(5)
    ]

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets[
            train_split_name] if training_args.do_train else None,
        eval_dataset=vectorized_datasets[
            eval_split_name] if training_args.do_eval else None,
        tokenizer=feature_extractor,
        **trainer_kwargs,
        callbacks=[PrintSamplesPredictionCallback(
            processor=processor,
            eval_dataset=samples_to_log)] if data_args.print_samples and training_args.do_eval else None,
    )

    # 8. Finally, we can start training

    # Training
    if training_args.do_train:

        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets[train_split_name])
        )
        metrics["train_samples"] = min(
            max_train_samples, len(vectorized_datasets[train_split_name])
        )

        trainer.log_metrics(train_split_name, metrics)
        trainer.save_metrics(train_split_name, metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(vectorized_datasets[eval_split_name])
        )
        metrics["eval_samples"] = min(max_eval_samples,
                                      len(vectorized_datasets[eval_split_name]))

        trainer.log_metrics(eval_split_name, metrics)
        trainer.save_metrics(eval_split_name, metrics)

    # Write model card and (optionally) push to hub
    config_name = (
        data_args.dataset_config_name
        if data_args.dataset_config_name is not None
        else "na"
    )
    kwargs = {
        "language": "he",
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "speech-recognition",
        "tags": ["automatic-speech-recognition", "robust-speech-event", "he"],
        "dataset_args": f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split: {data_args.eval_split_name}",
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()
