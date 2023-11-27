
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os

class EvaluationPipeline:
    """
    The pipeline for evaluating the quality of a summary according to the original text.
    """

    def __init__(self) -> None:
        pass