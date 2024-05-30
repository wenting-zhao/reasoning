import datasets
import streamlit as st

st.write("# Visualize code")

dataset = datasets.load_dataset(
    "json",
    data_files="out/model-a-samples-apps-train-gpt-4o-num8-start0-end128.json",
)

split = st.radio("split", options=dataset.keys())
dataset = dataset[split]

idx = st.number_input("Example number", min_value=0, max_value=len(dataset))
example = dataset[idx]

st.write("## Question")
st.write(example["question"])

st.write(example["is_correct"])

attempt_idx = st.number_input("attempt number", min_value=0, max_value=len(example["plan"]))

st.write("## Plan")
st.code(example["plan"][attempt_idx])

st.write("## Code")
st.code(example["code"][attempt_idx])
