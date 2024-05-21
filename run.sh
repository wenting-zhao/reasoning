model=$1
data=$2
name=$3
lr=$4
gs=$5
bs=4

python run_clm_no_trainer.py \
    --model_name_or_path $model \
    --train_file $data \
    --validation_file data/math/competition_math-test-cot.json \
    --per_device_train_batch_size $bs \
    --num_train_epochs 1 \
    --gradient_accumulation_steps $gs \
    --learning_rate $lr \
    --output_dir models/$name-lr$lr-bs$((bs*gs)) \
    --with_tracking \
    --report_to wandb \
    --low_cpu_mem_usage
