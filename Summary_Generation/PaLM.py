from json import load
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
import google.generativeai
from tqdm import tqdm
from time import sleep
from dotenv import load_dotenv
import os

def get_PaLM_result(prompt, google_api_key):
    llm = GooglePalm(google_api_key=google_api_key)
    llm.temperature = 0
    
    prompts = [prompt]
    llm_result = llm.generate(prompts=prompts)
    
    return llm_result.generations[0][0].text


