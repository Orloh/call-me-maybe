import pytest
from src.trie.trie import PrefixTrie
from src.automata import JSONPushdownAutomaton, ExactMatchFSM
from src.dfs.dfs import find_allowed_tokens

@pytest.fixture
def mock_schema():
    return {"name": lambda: ExactMatchFSM(["get_weather"])}

def test_dfs_finds_valid_tokens(mock_schema):
    """
    Proves that the DFS correctly traverses the Trie, clones the PDA, 
    and only returns tokens that are both in the Trie AND allowed by the PDA.
    """
    trie = PrefixTrie()
    trie.insert("get_weather", 1)
    trie.insert("get_money", 2)
    
    pda = JSONPushdownAutomaton(mock_schema)
    pda.advance("{")
    pda.advance('"')
    pda.advance('n')
    pda.advance('a')
    pda.advance('m')
    pda.advance('e')
    pda.advance('"')
    pda.advance(':')
    pda.advance('"')
    
    allowed = find_allowed_tokens(trie.root, pda)
    
    assert 1 in allowed
    assert 2 not in allowed

def test_dfs_prunes_invalid_branches(mock_schema):
    """
    Proves that if a Trie path is valid but the PDA rejects it, 
    the DFS prunes the branch.
    """
    trie = PrefixTrie()
    trie.insert("x_money", 3)
    
    pda = JSONPushdownAutomaton(mock_schema)
    pda.advance("{")
    pda.advance('"')
    pda.advance('n')
    pda.advance('a')
    pda.advance('m')
    pda.advance('e')
    pda.advance('"')
    pda.advance(':')
    pda.advance('"')
    
    allowed = find_allowed_tokens(trie.root, pda)
    assert 3 not in allowed
