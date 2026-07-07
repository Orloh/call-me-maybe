class TrieNode:
    """
    A single node in the Prefix Trie representing a single character.
    """
    def __init__(self):
        self.children: dict[str, 'TrieNode'] = {}
        self.token_id: int | None = None


class PrefixTrie:
    """
    A high-speed Prefix Trie for storing the LLM's entire vocabulary.
    Allows for extremely fast character-by-character token validation
    """
    def __init__(self):
        self.root = TrieNode()
        self.size = 0

    def insert(self, token_str: str, token_id: int) -> None:
        """
        Inserts a token string and tis correspondint ID into the Trie.
        """
        pass

    def build_from_vocabulary(self, vocabulary: dict[str, int]) -> None:
        """
        Utility method to ingest a massive dictionary of tokens in IDs all at once.
        """
        pass
