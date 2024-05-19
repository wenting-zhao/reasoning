model=meta-llama/Meta-Llama-3-8B-Instruct
data=data/math/competition_math-train-cot.json
name=Meta-Llama-3-8B-Instruct-cot
#lr=2e-6
#bs=128

for lr in 1e-6 2e-6 5e-6 1e-5 2e-5
    do
        for bs in 8 16 32 64 128 256
            do
                bash run.sh $model $data $name $lr $bs
	    done
    done
