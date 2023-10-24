from re import L
import pandas as pd
import numpy as np
import tqdm as tqdm
import PaLM
import os
from PaLM import get_PaLM_result
from dotenv import load_dotenv

def get_prompt_positive(text):
    template = f"""
You are a compliance officer who works at a financial institution. You need to create a summary that covers all important points. Summarize the following text.
```{text}```
SUMMARY:
"""
            
    return template

def get_prompt_negative(text):
    template = f"""
You are a compliance officer who works at a financial institution. However, this time, instead of producing an accurate summary, create a summary that is deliberately incorrect or unrelated while still sounding plausible. Summarize the following text.
```{text}```
FALSIFIED SUMMARY:
"""

    return template

def get_prompt_falsify(reference_summary):
    template = f"""
Given the reference summary, produce a plausible but incorrect summary.
```{reference_summary}```
SUMMARY:
"""
    return template

def get_positive_summary(text, model = 'PaLM'):
    if model == 'PaLM':
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        prompt = get_prompt_positive(text)
        result = get_PaLM_result(prompt, google_api_key)
    return result

def get_negative_summary(text, model = 'PaLM'):
    if model == 'PaLM':
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        prompt = get_prompt_negative(text, google_api_key)
        result = get_PaLM_result(prompt)
    return result




df = pd.read_csv("../Source_Text/reference_summary_original_text.csv")
text = df['text_extracted'][2]
    
    
positive_result = get_positive_summary(text)
negative_result = get_negative_summary(text)
    
with open("./testing/positive_result.txt", "w") as f:
    f.write(positive_result)
    
with open("./testing/negative_result.txt", "w") as f:
    f.write(negative_result)
    
reference_summary = df['Enforcement Summary'][2]
with open("./testing/reference_summary.txt", "w") as f:
    f.write(reference_summary)
    