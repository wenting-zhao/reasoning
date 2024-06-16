import datasets
import streamlit as st
import json

st.write("# Visualize code")

dataset = datasets.load_dataset(
    "json",
    data_files="out/model-a-samples-apps-train-gpt-4o-nofsFalse-num8-start0-end128.json",
)

split = st.radio("split", options=dataset.keys())
dataset = dataset[split]

idx = st.number_input("Example number", min_value=0, max_value=len(dataset))
example = dataset[idx]

st.write("## Difficulty:", example["difficulty"])

st.write("## Question")
st.write(example["question"].replace("\n", "  \n"))
st.text(example["question"])

st.write("## Attempts with correct answers")
st.write(example["is_correct"])

st.write("# Visualize attempt number")
attempt_idx = st.number_input("attempt number", min_value=0, max_value=len(example["plan"]))

st.write("## Plan")
st.write(example["plan"][attempt_idx])
st.text(example["plan"][attempt_idx])

st.write("## Code")
st.code(example["code"][attempt_idx])
st.text(example["code"][attempt_idx])


st.write("## IO")
st.code(json.loads(example["input_output"]))

st.write("## Solution")
st.code(json.loads(example["solutions"])[0])

