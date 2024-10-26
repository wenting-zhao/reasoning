import streamlit as st
import re

st.set_page_config(layout="wide")

def replacements(x):
    return (x
        .replace("\\begin{align*}", "$\\begin{align*}")
        .replace("\\end{align*}", "\\end{align*}$")
        .replace("\\[", "$")
        .replace("\\]", "$")
        .replace("\\(", "$")
        .replace("\\)", "$")
        .replace("$\$$", "\$")
        .replace("\n", "  \n")
    )


def split_text_and_keep_equations(text):
    # Define a regex pattern to identify LaTeX equations environments
    equation_pattern = r'(\\begin{align\*}.*?\\end{align\*})'
    equation_pattern = r'(\\\[.*?\\\])'
    equation_pattern = r'(\\\(.*?\\\))'
    #equation_pattern = r'(\\\\\[.*?\\\\\])|(\\\\\(.*?\\\\\))'
    
    # Use re.split to split the text while keeping the delimiters (equations)
    split_text_with_equations = re.split(equation_pattern, text, flags=re.DOTALL)

    return split_text_with_equations


answer = """
1. Let \\(y = \\arctan \\frac{1}{x}\\) and \\(z = \\arctan \\frac{1}{x^3}\\). This means \\(x = \\tan y\\) and \\(x^3 = \\tan z\\).\nWe know that \\(\\tan(\\alpha + \\beta) = \\frac{\\tan \\alpha + \\tan \\beta}{1 - \\tan \\alpha \\tan \\beta}\\). \nThus, \\(\\tan(y+z) = \\frac{\\tan y + \\tan z}{1 - \\tan y \\tan z} = \\frac{x + x^3}{1 - x^2} = 1\\).\nSolving this equation, we get \\(x^2 + x - 1 = 0\\) which gives \\(x = \\frac{-1 \\pm \\sqrt{5}}{2}\\).\nSince \\(x = \\tan y\\), we must have \\(x > 0\\) so \\(x = \\frac{-1 + \\sqrt{5}}{2}\\).\nNow plugging this value back into the original equation, we get\n\\[\\arctan \\frac{1}{\\frac{-1 + \\sqrt{5}}{2}} + \\arctan \\frac{1}{\\left(\\frac{-1 + \\sqrt{5}}{2}\\right)^3} = \\frac{\\pi}{4} \\rightarrow \\boxed{\\frac{\\pi}{4}}.\\]\n\n2. Let \\(\\alpha = \\frac{1}{x}\\) and \\(\\beta = \\frac{1}{x^3}\\). This means \\(x = \\frac{1}{\\alpha}\\) and \\(x^3 = \\frac{1}{\\beta}\\). \nThe relation given simplifies to \\(\\arctan \\alpha + \\arctan \\beta = \\frac{\\pi}{4}\\).\nUsing the identity \\(\\tan(\\alpha + \\beta) = \\frac{\\tan \\alpha + \\tan \\beta}{1 - \\tan \\alpha \\tan \\beta}\\), we have \\(\\alpha \\beta = 1\\).\nThus, we get \\(\\frac{1}{x} \\cdot \\frac{1}{x^3} = 1\\) which simplifies to \\(x^4 = 1\\) giving \\(x = 1\\).\nSubstitute into original equation:\n\\[\\arctan 1 + \\arctan 1 = \\frac{\\pi}{4} \\rightarrow \\boxed{\\frac{\\pi}{4}}.\\]\n\n3. Let \\(y = \\arctan \\frac{1}{x}\\) and \\(z = \\arctan \\frac{1}{x^3}\\). This implies \\(x = \\frac{1}{\\tan y}\\) and \\(x^3 = \\frac{1}{\\tan z}\\).\nUsing the trigonometric identity, we get \\(\\tan(y+z) = \\frac{\\tan y + \\tan z}{1 - \\tan y \\tan z} = 1\\).\nThus, we have \\(\\tan(y+z) = 1\\) implies \\(y + z = \\frac{\\pi}{4}\\).\nSubstitute into the original equation:\n\\[\\arctan \\frac{1}{x} + \\arctan \\frac{1}{x^3} = \\frac{\\pi}{4} \\rightarrow \\boxed{\\frac{\\pi}{4}}.\\]\n\n4. Let \\(y = \\arctan \\frac{1}{x}\\) and \\(z = \\arctan \\frac{1}{x^3}\\). This results \\(x = \\frac{1}{\\tan y}\\) and \\(x^3 = \\frac{1}{\\tan z}\\).\nUsing the trigonometric identity for arctan, we have \\(\\tan( \\arctan a + \\arctan b) = \\frac{a + b}{1 - ab}\\).\nTherefore, \\(\\frac{1}{x} \\cdot \\frac{1}{x^3} = 1\\) simplifies to \\(x^4 = 1\\) giving \\(x = 1\\).\nSubstitute into the original equation:\n\\[\\arctan 1 + \\arctan 1 = \\frac{\\pi}{4} \\rightarrow \\boxed{\\frac{\\pi}{4}}.\\]
"""

st.write(replacements(answer))
