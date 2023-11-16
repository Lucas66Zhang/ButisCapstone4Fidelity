import stanza
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os

class SummaryGrader:
    def __init__(self):
        self._nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma, ner')
        self._model = SentenceTransformer('bert-base-nli-mean-tokens')
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _split_text(self, text:str)->list:
        """
        Split text into sentences
        Args:
            text: the text to be split

        Returns:
            a list of sentences
        """
        doc = self._nlp(text)
        return [sentence.text for sentence in doc.sentences]

    def _sentence2embedding(self, sentences: list[str]) -> np.ndarray:
        """
        Convert sentences to embeddings
        Args:
            sentences: a list of sentences

        Returns:
            a matrix of embeddings, each row is an embedding
        """
        embeddings = self._model.encode(sentences)
        return embeddings

    def _cosine_similarity(self, embed_text: np.ndarray, embed_summary: np.ndarray) -> np.ndarray:
        """
        Calculate the cosine similarities between sentences of summary and sentences of text
        Args:
            embed_text: embedding matrix of text sentences
                        each row is an embedding
            embed_summary: embedding matrix of summary sentences
                        each row is an embedding

        Returns:
            a matrix of cosine similarities
        """

        dot_prod = embed_summary @ embed_text.T  # [i,j] is the dot product of summary sentence i and text sentence j
        norm = np.linalg.norm(embed_summary, axis=1) @ np.linalg.norm(embed_text, axis=1).T  # [i,j] is the norm of summary sentence i and text sentence j
        return dot_prod / norm

    def _topk_related(sim_matrix: np.ndarray, k: int) -> np.ndarray:
        """
        Find the indices of top k related sentences in text for each sentence in summary
        Args:
            sim_matrix: cosine similarity matrix
            k: number of sentences to be selected

        Returns:
            a matrix of indices
        """
        return sim_matrix.argsort(axis=1)[:, -k:]

    def _checker(self, sens_text: list[str], sen_summary: str) -> bool:
        """
        Check if the sentence from the summary con be obtained from the sentence from the text.
        Args:
            sens_text: list of sentences from the text
            sen_summary: the sentence from the summary

        Returns:
            a tuple of (bool, float)
            bool: True if the sentence from the summary can be obtained from the sentence from the text
            float: the probability that the sentence from the summary can be obtained from the sentence from the text
                True: >0.5
                False: <0.5
        """

        source_text = ''.join(sens_text)

        prompt = f"""
        As a compliance officer at a financial institution, you're tasked with evaluating the accuracy of a summary sentence based on its alignment with source sentences from a financial document. Consider the following criteria carefully:

        1. The summary accurately reflects the content of the source sentences, especially numerical information.
        2. All named entities in the summary are present in the source sentences.
        3. Relationships between entities in the summary are consistent with those in the source sentences.
        4. The directional flow of relationships among named entities matches between the summary and source sentences.
        5. There are no factual discrepancies between the summary and source sentences.
        6. The summary does not introduce any entities not found in the source sentences.

        Your job is to determine if the summary adheres to these criteria. Answer "Yes" if it does, or "No" if it doesn't.

        Summary sentence: ```{sen_summary}```

        Source sentences: ```{source_text}```

        Final Answer (Yes/No only): 
        """

        response = self._client.chat.completions.create(
            model='gpt-4',
            messages=[{'role': "user", 'content': prompt}],
            max_tokens=1
        )

        res = response.choices[0].text.lower()
        if res == 'yes':
            return True
        elif res == 'no':
            return False
        else:
            raise ValueError("Invalid response from OpenAI API")

    def evaluate(self, text:str, summary:str, k:int) -> float:
        """
        evaluate the quality of the summary according to the given text
        Args:
            text: original text
            summary: summary to be evaluated
            k: number of sentences to be selected from the text

        Returns:
            a float number between 0 and 1, the higher the better
        """

        # split the text into sentences
        sens_text = self._split_text(text)
        # split the summary into sentences
        sens_summary = self._split_text(summary)

        # convert sentences to embeddings
        embed_text = self._sentence2embedding(sens_text)
        embed_summary = self._sentence2embedding(sens_summary)

        # calculate cosine similarity
        sim_matrix = self._cosine_similarity(embed_text, embed_summary)

        # find top k related sentences
        topk = self._topk_related(sim_matrix, k)

        # check if the sentence from the summary can be obtained from the sentence from the text
        denominator = 0
        numerator = 0
        for idx, sen in enumerate(sens_summary):
            sens_text_selected = [sens_text[i] for i in topk[idx]]
            res = self._checker(sens_text_selected, sen)
            if res:
                numerator += 1
            denominator += 1
        return numerator / denominator

