import pytest
from src.trie import PrefixTrie

def test_trie_basic_insertion():
    """
    Verifies that a single token is correctly inserted into the Trie
    and that the path to the leaf node is properly established.
    """
    trie = PrefixTrie()
    trie.insert("cat", 1)
    
    # Traverse manually to verify structure
    node = trie.root
    for char in "cat":
        assert char in node.children
        node = node.children[char]
    
    assert node.token_id == 1
    assert trie.size == 1

def test_trie_structural_integrity():
    """
    Verifies the entire internal structure of the Trie, ensuring 
    prefix sharing is correctly implemented for 'cat' and 'car'.
    """
    trie = PrefixTrie()
    trie.insert("cat", 1)
    trie.insert("car", 2)
    
    assert len(trie.root.children) == 1
    assert 'c' in trie.root.children
    
    node_c = trie.root.children['c']
    assert len(node_c.children) == 1
    assert 'a' in node_c.children
    
    node_a = node_c.children['a']
    assert len(node_a.children) == 2
    assert 't' in node_a.children
    assert 'r' in node_a.children
    
    node_t = node_a.children['t']
    node_r = node_a.children['r']
    
    assert node_t.token_id == 1
    assert len(node_t.children) == 0
    
    assert node_r.token_id == 2
    assert len(node_r.children) == 0

def test_trie_empty_and_duplicate():
    """
    Checks the robustness of insertion logic:
    1. Empty strings should be ignored.
    2. Duplicate insertions of the same token string should not 
       overwrite the existing token_id.
    """
    trie = PrefixTrie()
    trie.insert("", 1)
    assert trie.size == 0

    trie.insert("dog", 3)
    trie.insert("dog", 4)

    node = trie.root.children['d'].children['o'].children['g']
    assert node.token_id == 3 
    assert trie.size == 1

def test_build_from_vocab():
    """
    Validates the batch ingestion logic, ensuring multiple tokens are 
    correctly mapped and stored in the Trie.
    """
    trie = PrefixTrie()
    vocab = {"apple": 1, "banana": 2, "app": 3}
    trie.build_from_vocab(vocab)
    
    assert trie.size == 3
    assert trie.root.children['a'].children['p'].children['p'].token_id == 3
    assert trie.root.children['a'].children['p'].children['p'].children['l'].children['e'].token_id == 1
