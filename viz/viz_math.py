import datasets
import streamlit as st
import re

st.set_page_config(layout="wide")

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


st.write("# Visualize math")

prior_dataset = datasets.load_dataset(
    "json",
    data_files="data/math/star_prior_iteration0_max1.json",
)["train"]
posterior_dataset = datasets.load_dataset(
    "json",
    data_files="data/math/star_posterior_iteration0v2_max1.json",
)["train"]


#st.text(replacements(question))


col1, col2 = st.columns(2)
with col1:
    idx1 = st.number_input("Prior example number", min_value=0, max_value=len(prior_dataset))
    prior_example = prior_dataset[idx1]["text"]

    st.write("## Question")
    question = prior_example[0]["content"]
    st.write(replacements(question))

    st.write("## Prior answer")
    prior_answer = prior_example[1]["content"]

    blocks = split_text_and_keep_equations(prior_answer)
    for block in blocks:
        if "align*" in block:
            st.latex(block)
        else:
            st.write(replacements(block))

with col2:
    idx2 = st.number_input("Posterior example number", min_value=0, max_value=len(prior_dataset))
    posterior_example = posterior_dataset[idx2]["text"]
    st.write("## Question")
    question = posterior_example[0]["content"]
    st.write(replacements(question))

    st.write("## Posterior answer")
    posterior_answer = posterior_example[1]["content"]

    blocks = split_text_and_keep_equations(posterior_answer)
    for block in blocks:
        if "align*" in block:
            st.latex(block)
        else:
            st.write(replacements(block))
