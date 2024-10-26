from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import nltk
import matplotlib.pyplot as plt
import sys

ds = load_dataset("json", data_files=sys.argv[1:], split="train")

key = 'solution' #'output' if 'output' in ds.column_names else 'star'

ds = ds.to_list()
#words = [xx for x in ds for xx in x[key]]
words = [x[key] for x in ds]
count = []
for each in tqdm(words):
    #curr = len(nltk.word_tokenize(each))
    curr = len(each.split())
    count.append(curr)
print(np.mean(count), np.std(count))

num_bins = 100
 
plt.figure(figsize=(4,5))
n, bins, patches = plt.hist(count, num_bins)
plt.xlabel('Length', weight="bold")
plt.ylabel('#z', weight="bold")

plt.savefig('len-dist.pdf', bbox_inches='tight')
