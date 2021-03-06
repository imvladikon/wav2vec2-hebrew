export CUDA_VISIBLE_DEVICES="0,1"

python -m torch.distributed.launch --nproc_per_node=2 run_train.py \
	--dataset_name="imvladikon/hebrew_speech_common" \
	--use_auth_token="$HUGGINGFACE_API_TOKEN" \
    --audio_column_name="audio" \
    --text_column_name="sentence" \
	--model_name_or_path="facebook/wav2vec2-xls-r-1b" \
	--output_dir="./wav2vec2-xls-r-300m-hebrew" \
	--overwrite_output_dir \
	--evaluation_strategy="steps" \
	--length_column_name="input_length" \
	--gradient_checkpointing \
	--fp16 \
	--group_by_length \
	--num_train_epochs="10" \
	--per_device_train_batch_size="6" \
	--per_device_eval_batch_size="6" \
	--gradient_accumulation_steps="4" \
	--learning_rate="3e-4" \
	--warmup_steps="1000" \
	--save_steps="1000" \
	--eval_steps="1000" \
	--preprocessing_num_workers="$(nproc)" \
	--logging_steps="2000" \
	--layerdrop="0.0" \
	--activation_dropout="0.1" \
	--save_total_limit="3" \
	--freeze_feature_encoder \
	--feat_proj_dropout="0.0" \
	--mask_time_prob="0.75" \
	--mask_time_length="10" \
	--mask_feature_prob="0.25" \
	--mask_feature_length="64" \
	--do_train --do_eval \
	--print_samples \
	--use_augmentations