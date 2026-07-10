import copy
from src.trie import TrieNode
from src.automata import JSONPushdownAutomaton

def find_allowed_tokens(node: TrieNode, pda: JSONPushdownAutomaton) -> list[int]:
    """
    Recursively explores the Trie to find all token IDs that satisfy the PDA's constraints.
    """
    valid_tokens: list[int] = []

    if node.token_id is not None:
        valid_tokens.append(node.token_id)

    for char, child_node in node.children.items():
        pda_clone = copy.deepcopy(pda)

        if pda_clone.advance(char):
            child_tokens = find_allowed_tokens(child_node, pda_clone)
            valid_tokens.extend(child_tokens)

    return valid_tokens
