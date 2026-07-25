from src.trie import TrieNode
from src.automata import JSONPushdownAutomaton


def find_allowed_tokens(
    node: TrieNode,
    pda: JSONPushdownAutomaton
) -> list[int]:
    """
    Recursively explores the Trie to find all token IDs that satisfy
    the PDA's constraints.

    Each edge is pre-filtered with pda.accepts_char() — a cheap,
    non-mutating check — so clones are only created for branches the
    PDA can actually accept. The clone's advance() remains the final
    arbiter: a branch is explored only if the char is truly consumed.
    """
    valid_tokens: list[int] = []

    if node.token_id is not None:
        valid_tokens.append(node.token_id)

    for char, child_node in node.children.items():
        if not pda.accepts_char(char):
            continue

        pda_clone = pda.clone()

        if pda_clone.advance(char):
            child_tokens = find_allowed_tokens(child_node, pda_clone)
            valid_tokens.extend(child_tokens)

    return valid_tokens
