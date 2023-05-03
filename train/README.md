
# wav2vec2-xls-r-300m-hebrew

A fine-tuned versions of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on the private datasets in 2 stages:
* firstly was fine-tuned on a small dataset with good samples 
* then the obtained model was fine-tuned on a large dataset with the small good dataset, with various samples from different sources, and with an unlabeled dataset that was weakly labeled using a previously trained model.


## Weights

* [imvladikon/wav2vec2-xls-r-300m-hebrew](https://huggingface.co/imvladikon/wav2vec2-xls-r-300m-hebrew)
* [imvladikon/wav2vec2-xls-r-300m-lm-hebrew](https://huggingface.co/imvladikon/wav2vec2-xls-r-300m-lm-hebrew)

## Wandb

[wandb logs](https://wandb.ai/imvladikon/wav2vec2-hebrew?workspace=user-imvladikon)

## Usage

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

## Datasets descriptions

Small dataset:

| split  |size(gb) | n_samples | duration(hrs)|  
|---|---|---|---|
|train|4.19| 20306  | 28 | 
|dev  |1.05|  5076 |  7 |

Large dataset: 

| split  |size(gb) | n_samples | duration(hrs)|
|---|---|---|---|
|train|12.3| 90777  | 69  |
|dev  |1.05|  20246 |  14* |

(*weakly labeled data wasn't used in validation set)

## Results

After firts training it achieves:

on small dataset
- Loss: 0.5438
- WER: 0.1773

on large dataset
- WER: 0.3811

after second training:
on small dataset
- WER: 0.1697

on large dataset
- Loss: 0.4502
- WER: 0.2318


## Training procedure

### Training hyperparameters


#### First training

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 4
- total_train_batch_size: 64
- total_eval_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1000
- num_epochs: 100.0
- mixed_precision_training: Native AMP

Training results

| Training Loss | Epoch | Step  | Validation Loss | Wer    |
|:-------------:|:-----:|:-----:|:---------------:|:------:|
| No log        | 3.15  | 1000  | 0.5203          | 0.4333 |
| 1.4284        | 6.31  | 2000  | 0.4816          | 0.3951 |
| 1.4284        | 9.46  | 3000  | 0.4315          | 0.3546 |
| 1.283         | 12.62 | 4000  | 0.4278          | 0.3404 |
| 1.283         | 15.77 | 5000  | 0.4090          | 0.3054 |
| 1.1777        | 18.93 | 6000  | 0.3893          | 0.3006 |
| 1.1777        | 22.08 | 7000  | 0.3968          | 0.2857 |
| 1.0994        | 25.24 | 8000  | 0.3892          | 0.2751 |
| 1.0994        | 28.39 | 9000  | 0.4061          | 0.2690 |
| 1.0323        | 31.54 | 10000 | 0.4114          | 0.2507 |
| 1.0323        | 34.7  | 11000 | 0.4021          | 0.2508 |
| 0.9623        | 37.85 | 12000 | 0.4032          | 0.2378 |
| 0.9623        | 41.01 | 13000 | 0.4148          | 0.2374 |
| 0.9077        | 44.16 | 14000 | 0.4350          | 0.2323 |
| 0.9077        | 47.32 | 15000 | 0.4515          | 0.2246 |
| 0.8573        | 50.47 | 16000 | 0.4474          | 0.2180 |
| 0.8573        | 53.63 | 17000 | 0.4649          | 0.2171 |
| 0.8083        | 56.78 | 18000 | 0.4455          | 0.2102 |
| 0.8083        | 59.94 | 19000 | 0.4587          | 0.2092 |
| 0.769         | 63.09 | 20000 | 0.4794          | 0.2012 |
| 0.769         | 66.25 | 21000 | 0.4845          | 0.2007 |
| 0.7308        | 69.4  | 22000 | 0.4937          | 0.2008 |
| 0.7308        | 72.55 | 23000 | 0.4920          | 0.1895 |
| 0.6927        | 75.71 | 24000 | 0.5179          | 0.1911 |
| 0.6927        | 78.86 | 25000 | 0.5202          | 0.1877 |
| 0.6622        | 82.02 | 26000 | 0.5266          | 0.1840 |
| 0.6622        | 85.17 | 27000 | 0.5351          | 0.1854 |
| 0.6315        | 88.33 | 28000 | 0.5373          | 0.1811 |
| 0.6315        | 91.48 | 29000 | 0.5331          | 0.1792 |
| 0.6075        | 94.64 | 30000 | 0.5390          | 0.1779 |
| 0.6075        | 97.79 | 31000 | 0.5459          | 0.1773 |

#### Second training

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 4
- total_train_batch_size: 64
- total_eval_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1000
- num_epochs: 60.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Wer    |
|:-------------:|:-----:|:-----:|:---------------:|:------:|
| No log        | 0.7   | 1000  | 0.5371          | 0.3811 |
| 1.3606        | 1.41  | 2000  | 0.5247          | 0.3902 |
| 1.3606        | 2.12  | 3000  | 0.5126          | 0.3859 |
| 1.3671        | 2.82  | 4000  | 0.5062          | 0.3828 |
| 1.3671        | 3.53  | 5000  | 0.4979          | 0.3672 |
| 1.3421        | 4.23  | 6000  | 0.4906          | 0.3816 |
| 1.3421        | 4.94  | 7000  | 0.4784          | 0.3651 |
| 1.328         | 5.64  | 8000  | 0.4810          | 0.3669 |
| 1.328         | 6.35  | 9000  | 0.4747          | 0.3597 |
| 1.3109        | 7.05  | 10000 | 0.4813          | 0.3808 |
| 1.3109        | 7.76  | 11000 | 0.4631          | 0.3561 |
| 1.2873        | 8.46  | 12000 | 0.4603          | 0.3431 |
| 1.2873        | 9.17  | 13000 | 0.4579          | 0.3533 |
| 1.2661        | 9.87  | 14000 | 0.4471          | 0.3365 |
| 1.2661        | 10.58 | 15000 | 0.4584          | 0.3437 |
| 1.249         | 11.28 | 16000 | 0.4461          | 0.3454 |
| 1.249         | 11.99 | 17000 | 0.4482          | 0.3367 |
| 1.2322        | 12.69 | 18000 | 0.4464          | 0.3335 |
| 1.2322        | 13.4  | 19000 | 0.4427          | 0.3454 |
| 1.22          | 14.1  | 20000 | 0.4440          | 0.3395 |
| 1.22          | 14.81 | 21000 | 0.4459          | 0.3378 |
| 1.2044        | 15.51 | 22000 | 0.4406          | 0.3199 |
| 1.2044        | 16.22 | 23000 | 0.4398          | 0.3155 |
| 1.1913        | 16.92 | 24000 | 0.4237          | 0.3150 |
| 1.1913        | 17.63 | 25000 | 0.4287          | 0.3279 |
| 1.1705        | 18.34 | 26000 | 0.4253          | 0.3103 |
| 1.1705        | 19.04 | 27000 | 0.4234          | 0.3098 |
| 1.1564        | 19.75 | 28000 | 0.4174          | 0.3076 |
| 1.1564        | 20.45 | 29000 | 0.4260          | 0.3160 |
| 1.1461        | 21.16 | 30000 | 0.4235          | 0.3036 |
| 1.1461        | 21.86 | 31000 | 0.4309          | 0.3055 |
| 1.1285        | 22.57 | 32000 | 0.4264          | 0.3006 |
| 1.1285        | 23.27 | 33000 | 0.4201          | 0.2880 |
| 1.1135        | 23.98 | 34000 | 0.4131          | 0.2975 |
| 1.1135        | 24.68 | 35000 | 0.4202          | 0.2849 |
| 1.0968        | 25.39 | 36000 | 0.4105          | 0.2888 |
| 1.0968        | 26.09 | 37000 | 0.4210          | 0.2834 |
| 1.087         | 26.8  | 38000 | 0.4123          | 0.2843 |
| 1.087         | 27.5  | 39000 | 0.4216          | 0.2803 |
| 1.0707        | 28.21 | 40000 | 0.4161          | 0.2787 |
| 1.0707        | 28.91 | 41000 | 0.4186          | 0.2740 |
| 1.0575        | 29.62 | 42000 | 0.4118          | 0.2845 |
| 1.0575        | 30.32 | 43000 | 0.4243          | 0.2773 |
| 1.0474        | 31.03 | 44000 | 0.4221          | 0.2707 |
| 1.0474        | 31.73 | 45000 | 0.4138          | 0.2700 |
| 1.0333        | 32.44 | 46000 | 0.4102          | 0.2638 |
| 1.0333        | 33.15 | 47000 | 0.4162          | 0.2650 |
| 1.0191        | 33.85 | 48000 | 0.4155          | 0.2636 |
| 1.0191        | 34.56 | 49000 | 0.4129          | 0.2656 |
| 1.0087        | 35.26 | 50000 | 0.4157          | 0.2632 |
| 1.0087        | 35.97 | 51000 | 0.4090          | 0.2654 |
| 0.9901        | 36.67 | 52000 | 0.4183          | 0.2587 |
| 0.9901        | 37.38 | 53000 | 0.4251          | 0.2648 |
| 0.9795        | 38.08 | 54000 | 0.4229          | 0.2555 |
| 0.9795        | 38.79 | 55000 | 0.4176          | 0.2546 |
| 0.9644        | 39.49 | 56000 | 0.4223          | 0.2513 |
| 0.9644        | 40.2  | 57000 | 0.4244          | 0.2530 |
| 0.9534        | 40.9  | 58000 | 0.4175          | 0.2538 |
| 0.9534        | 41.61 | 59000 | 0.4213          | 0.2505 |
| 0.9397        | 42.31 | 60000 | 0.4275          | 0.2565 |
| 0.9397        | 43.02 | 61000 | 0.4315          | 0.2528 |
| 0.9269        | 43.72 | 62000 | 0.4316          | 0.2501 |
| 0.9269        | 44.43 | 63000 | 0.4247          | 0.2471 |
| 0.9175        | 45.13 | 64000 | 0.4376          | 0.2469 |
| 0.9175        | 45.84 | 65000 | 0.4335          | 0.2450 |
| 0.9026        | 46.54 | 66000 | 0.4336          | 0.2452 |
| 0.9026        | 47.25 | 67000 | 0.4400          | 0.2427 |
| 0.8929        | 47.95 | 68000 | 0.4382          | 0.2429 |
| 0.8929        | 48.66 | 69000 | 0.4361          | 0.2415 |
| 0.8786        | 49.37 | 70000 | 0.4413          | 0.2398 |
| 0.8786        | 50.07 | 71000 | 0.4392          | 0.2415 |
| 0.8714        | 50.78 | 72000 | 0.4345          | 0.2406 |
| 0.8714        | 51.48 | 73000 | 0.4475          | 0.2402 |
| 0.8589        | 52.19 | 74000 | 0.4473          | 0.2374 |
| 0.8589        | 52.89 | 75000 | 0.4457          | 0.2357 |
| 0.8493        | 53.6  | 76000 | 0.4462          | 0.2366 |
| 0.8493        | 54.3  | 77000 | 0.4494          | 0.2356 |
| 0.8395        | 55.01 | 78000 | 0.4472          | 0.2352 |
| 0.8395        | 55.71 | 79000 | 0.4490          | 0.2339 |
| 0.8295        | 56.42 | 80000 | 0.4489          | 0.2318 |
| 0.8295        | 57.12 | 81000 | 0.4469          | 0.2320 |
| 0.8225        | 57.83 | 82000 | 0.4478          | 0.2321 |
| 0.8225        | 58.53 | 83000 | 0.4525          | 0.2326 |
| 0.816         | 59.24 | 84000 | 0.4532          | 0.2316 |
| 0.816         | 59.94 | 85000 | 0.4502          | 0.2318 |


### Framework versions

- Transformers 4.17.0.dev0
- Pytorch 1.10.2+cu102
- Datasets 1.18.2.dev0
- Tokenizers 0.11.0
