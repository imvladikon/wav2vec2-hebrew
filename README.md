
# Hebrew Speech Recognition with Wav2Vec2


## Usage

### Without package installation (using `transformers` library)


```python
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    AutoFeatureExtractor,
    Wav2Vec2ForCTC,
    AutoTokenizer
)

pretrained_model_name_or_path = "imvladikon/wav2vec2-xls-r-300m-hebrew"
asr = AutomaticSpeechRecognitionPipeline(
    feature_extractor=AutoFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path
    ),
    model=Wav2Vec2ForCTC.from_pretrained(
        pretrained_model_name_or_path
    ),
    tokenizer=AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path
    ))
filename = "audio.wav"
print(asr(filename))
```
Chunking file into smaller chunks is not implemented yet. 

### With package installation

```bash
pip install git+https://github.com/imvladikon/wav2vec2-hebrew
```

#### Speech recognition

```python

from wav2vec2_hebrew import HebrewSpeechRecognitionPipeline

asr = HebrewSpeechRecognitionPipeline()
filename = "./samples/013b882862704b7792ce16cd944b98470.wav"
output = asr(filename)
print(output)
```

#### Alignment
```python
import torchaudio
from wav2vec2_hebrew import HebrewWav2Vec2Aligner

filename = "bereshit011.wav"
text = "בראשית ברא אלוהים את השמיים ואת הארץ"
aligner = HebrewWav2Vec2Aligner(input_sample_rate=16000, use_cuda=True)
# aligning segments to text
segments = aligner.align_data(filename, text)[0]

# showing in IPython (notebook)
waveform, sample_rate = torchaudio.load(filename)
aligner.show_segments(waveform, segments)
```

## Training process

Training logs and details are available in the [train](train) folder.


### Weights

* [imvladikon/wav2vec2-xls-r-300m-hebrew](https://huggingface.co/imvladikon/wav2vec2-xls-r-300m-hebrew)
* [imvladikon/wav2vec2-xls-r-300m-lm-hebrew](https://huggingface.co/imvladikon/wav2vec2-xls-r-300m-lm-hebrew)

