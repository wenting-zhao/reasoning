model=$1
data=$2
name=$3
lr=$4
bs=$5

python run_clm_no_trainer.py \
    --model_name_or_path $model \
    --train_file $data \
    --validation_file data/math/competition_math-test-cot.json \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps $bs \
    --learning_rate $lr \
    --output_dir models/$name-lr$lr-bs$bs \
    --with_tracking \
    --report_to wandb \
    --low_cpu_mem_usage
