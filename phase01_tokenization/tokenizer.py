class SimpleTokenizer:

    def __init__(self, text):
        # Convert text into tokens
        tokens = text.lower().split()

        # Create vocabulary
        vocabulary = sorted(set(tokens))

        # Add special token for unknown words
        vocabulary.append("<UNK>")

        # token -> ID
        self.token_to_id = {
            token: index
            for index, token in enumerate(vocabulary)
        }

        # ID -> token
        self.id_to_token = {
            index: token
            for token, index in self.token_to_id.items()
        }

    def encode(self, text):
        """
        Convert text into token IDs.
        """

        tokens = text.lower().split()

        ids = []

        for token in tokens:

            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                ids.append(self.token_to_id["<UNK>"])

        return ids

    def decode(self, ids):
        """
        Convert token IDs back into text.
        """

        tokens = []

        for token_id in ids:
            tokens.append(self.id_to_token[token_id])

        return " ".join(tokens)