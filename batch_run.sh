for i in {0..7499..500}
    do
        python sample-model-a.py --dataset_name hendrycks/competition_math --dataset_split train --model meta-llama/Meta-Llama-3-8B --num_samples 30 --start $i --end $((${i}+500))
    done
