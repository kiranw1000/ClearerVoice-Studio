#!/bin/sh

#####
# Modify these lines
gpu_id=												# Visible GPUs
n_gpu=0														# Number of GPU used for training
checkpoint_dir=''									# Leave empty if it's a new training, otherwise provide the name as 'checkpoints/log_...'
config_pth=/Users/kiran/Documents_local/ASPIRE/ClearerVoice-Studio/train/target_speaker_extraction/config/config_KUL_eeg_neuroheed_2spk_contrastive.yaml		# The config file, only used if it's a new training
name=${1:-''}
#####

echo $name

# create checkpoint log folder
eval "$(conda shell.bash hook)"
conda activate aspire
if [ -z ${checkpoint_dir} ]; then
	checkpoint_dir='checkpoints/log_'$(date '+%Y-%m-%d(%H:%M:%S)')${name}
	train_from_last_checkpoint=0
	mkdir -p ${checkpoint_dir}
	cp $config_pth ${checkpoint_dir}/config.yaml
    echo "New training"
else
	train_from_last_checkpoint=1
	config_pth=${checkpoint_dir}/config.yaml
fi
yaml_name=log_$(date '+%Y-%m-%d(%H:%M:%S)')
echo $yaml_name
cat $config_pth > ${checkpoint_dir}/${yaml_name}.txt

echo $config_pth > ${checkpoint_dir}/${yaml_name}.txt

# call training
export PYTHONWARNINGS="ignore"
# CUDA_VISIBLE_DEVICES="$gpu_id" \
# python -W ignore \
# -m torch.distributed.launch \
# --nproc_per_node=$n_gpu \
# --master_port=$(date '+88%S') \
python -W ignore \
train.py \
--config $config_pth \
--checkpoint_dir $checkpoint_dir \
--train_from_last_checkpoint $train_from_last_checkpoint \
>>${checkpoint_dir}/$yaml_name.txt 2>&1
