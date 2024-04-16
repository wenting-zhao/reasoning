#for i in {0..7499..500}
#    do
#        sbatch --qos=normal --requeue -p gpu,rush -N1 -n1 --time=1000:00:00 --ntasks-per-core=4 --mem-per-cpu=32G --gres=gpu:a6000:1 --wrap "python sample-model-a.py --dataset_name hendrycks/competition_math --dataset_split train --model mistralai/Mistral-7B-v0.1 --num_samples 30 --start $i --end $((${i}+500))"
#    done
for f in out/model-a-samples-competition_math-train-Mistral-7B-v0.1-num30*
    do
        sbatch --qos=normal --requeue -p gpu,rush -N1 -n1 --time=1000:00:00 --ntasks-per-core=4 --mem-per-cpu=32G --gres=gpu:a6000:1 --wrap "python asymmetric_filtering.py --model_name mistralai/Mistral-7B-v0.1 --dataset_name $f"
    done
