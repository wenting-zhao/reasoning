import datasets

apps = datasets.load_dataset("codeparrot/apps")

num_train = sum(x["starter_code"] != "" for x in apps["train"])
num_test = sum(x["starter_code"] != "" for x in apps["test"])

print(num_train, num_test, "out of 5000")
import pdb; pdb.set_trace()
