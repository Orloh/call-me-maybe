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
        Inserts a token string and its corresponding ID into the Trie.
        """
        if not token_str:
            return

        current_node = self.root

        for char in token_str:
            if char not in current_node.children:
                current_node.children[char] = TrieNode()
            
            current_node = current_node.children[char]

        if current_node.token_id is None:
            current_node.token_id = token_id
            self.size += 1

    def build_from_vocab(self, vocabulary: dict[str, int]) -> None:
        """
        Utility method to ingest a massive dictionary of tokens and IDs all at once.
        """
        for token_str, token_id in vocabulary.items():
            self.insert(token_str, token_id)

