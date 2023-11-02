import stanza
import numpy as np
from sentence_transformers import SentenceTransformer

class SummaryGrader:
    def __init__(self):
        self._nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma, ner')
        self._model = SentenceTransformer('bert-base-nli-mean-tokens')

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

    def _checker(self, sens_text: list[str], sen_summary: str) -> (bool, float):
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

        # to be completed

        res = ____
        prob = ____

        return (res, prob)

    def evaluate(self, text: str, summary: str, k: int) -> float:
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
            res, _ = self._checker(sens_text_selected, sen)
            if res:
                numerator += 1
            denominator += 1
        return numerator / denominator

