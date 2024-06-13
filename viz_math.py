import datasets
import streamlit as st
import re


def replacements(x):
    return (x
        .replace("\\begin{align*}", "$\\begin{align*}")
        .replace("\\end{align*}", "\\end{align*}$")
        .replace("\\[", "$")
        .replace("\\]", "$")
        .replace("$\$$", "\$")
        .replace("\n", "  \n")
    )


def split_text_and_keep_equations(text):
    # Define a regex pattern to identify LaTeX equations environments
    equation_pattern = r'(\\begin{align\*}.*?\\end{align\*})'
    
    # Use re.split to split the text while keeping the delimiters (equations)
    split_text_with_equations = re.split(equation_pattern, text, flags=re.DOTALL)

    return split_text_with_equations


st.write("# Visualize code")

prior_dataset = datasets.load_dataset(
    "json",
    data_files="data/math/star_prior_iteration0_max1.json",
)["train"]
posterior_dataset = datasets.load_dataset(
    "json",
    data_files="data/math/star_posterior_iteration0v2_max1.json",
)["train"]

idx = st.number_input("Example number", min_value=0, max_value=len(prior_dataset))
prior_example = prior_dataset[idx]["text"]
posterior_example = posterior_dataset[idx]["text"]

st.write("## Question")
question = prior_example[0]["content"]
st.write(replacements(question))
#st.text(replacements(question))

st.write("## Prior answer")
prior_answer = prior_example[1]["content"]

blocks = split_text_and_keep_equations(prior_answer)
for block in blocks:
    if "align*" in block:
        st.latex(block)
    else:
        st.write(replacements(block))
