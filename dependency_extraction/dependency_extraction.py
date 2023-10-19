import stanza
class DependencyExtractor():

    def __init__(self, focused=True):
        """
        Initialize the dependency extractor.
        Parameters
        ----------
        focused: bool, optional, default: True, whether to only extract dependencies with content words.
        """
        self.extractor = stanza.Pipeline("en")
        self.focused = focused
        self.focused_set = None
        if focused:
            self.focused_set = {'NOUN', 'VERB', 'ADJ', 'ADV'}

    def extract(self, text, for_words=False, for_idx=True) -> list:
        """
        Extract the dependencies from the text.
        Parameters
        ----------
        text: str, the text to extract dependencies from.
        for_words: bool, optional, default: False, whether to return the dependencies in the form of words.
        for_idx: bool, optional, default: True, whether to return the dependencies in the form of indices.

        Returns
        -------
        list or lists, the list of dependencies.
        """
        if not for_words and not for_idx:
            # must extract dependencies in at least one form
            raise ValueError("At least one of for_words and for_idx must be True.")

        # extract the dependencies
        doc = self.extractor(text)

        # extract the dependencies in the form of words
        if for_words:
            dependencies_in_words = []

        # extract the dependencies in the form of indices
        if for_idx:
            dependencies_in_idx = []

        for sent in doc.sentences:
            for word in sent.words:
                # if the dependency is not focused, skip it
                if self.focused and word.upos not in self.focused_set:
                    continue
                if for_words:
                    dependencies_in_words.append((word.text, sent.words[word.head-1].text, word.deprel))
                if for_idx:
                    dependencies_in_idx.append((word.id, word.head, word.deprel))

        if for_words and for_idx:
            return dependencies_in_words, dependencies_in_idx
        elif for_words:
            return dependencies_in_words
        else:
            return dependencies_in_idx



