# reasoning

## Todos
* add pyproject.toml later. cpu and gpu versions.

Sample from LMs:
```
python prompting.py --model_name meta-llama/Llama-3.1-8B-Instruct --dataset_name hendrycks/competition_math --dataset_split train --start 0 --end 3750 --port 30000 --num_sample 32 --start_server
```

Get logprob from LMs:
```
python perplexity.py meta-llama/Llama-3.1-8B-Instruct --dataset_name <dataset json from the last step>
```

Generate a dataset:
```
python filter_by_posterior.py <dataset json for the last step>
```

Train on the dataset:
```
bash run.sh meta-llama/Llama-3.1-8B-Instruct <dataset json produced from the last step> prior_sample_ranked_by_posterior 1e-5 128
```
