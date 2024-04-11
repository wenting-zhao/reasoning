model=$1
data=$2
name=$3

python run_clm_no_trainer.py \
    --model_name_or_path $model \
    --train_file $data \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --output_dir models/$model-$name \
    --with_tracking \
    --report_to wandb \
    --low_cpu_mem_usage
