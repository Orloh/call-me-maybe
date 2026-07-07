class TrieNode:
    """
    A single node in the Prefix Trie representing a single character.
    """
    def __init__(self):
        self.children: dict[str, 'TrieNode'] = {}
        self.token_id: int | None = None
