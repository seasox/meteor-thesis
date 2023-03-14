from typing import List, Optional, Tuple, Dict, Union, Iterable

import torch
from transformers import GPT2Tokenizer


class TokenTrie:
    label: bytes
    token: int
    probability: Union[int, float]
    edges: Dict[bytes, 'TokenTrie']
    parent: Optional['TokenTrie']
    lookup: Dict[int, 'TokenTrie']

    def __init__(self, label: bytes = None, probability: Union[int, float] = None, token: int = None, edges: Dict = None, parent: 'TokenTrie' = None):
        from collections import OrderedDict
        if edges is None:
            edges = OrderedDict()
        self.label = label
        self.token = token
        self.probability = probability
        self.edges = OrderedDict(edges)  # label: node
        self.parent = parent
        self.lookup = {}

    @classmethod
    def from_tokenizer(cls, enc: GPT2Tokenizer) -> 'TokenTrie':
        labels = [(v, k.encode('utf-8', errors=enc.errors)) for k, v in enc.encoder.items()]
        return cls.from_labels(labels)

    @classmethod
    def from_labels(cls, labels: List[Tuple[int, bytes]]) -> 'TokenTrie':
        from collections import OrderedDict
        self = cls()
        self.label = None
        self.token = None
        self.edges = OrderedDict()
        self.parent = None
        self.lookup = {}
        for index, label in labels:
            self.lookup[index] = self.insert(label=label, token=index)
            if self.lookup[index] is None:
                raise Exception(f'insert inconsistency: {label}')
        return self

    def update(self, probabilities: Iterable[Tuple[torch.Tensor, int]]):
        for k, trie in self.lookup.items():
            trie.probability = None
        for label, prob in probabilities:
            self.lookup[label.item()].probability = prob

    def insert(self, label: bytes, token: int, probability=None) -> 'TokenTrie':
        assert token not in self.lookup
        assert isinstance(label, bytes)
        if token is None:
            token = label
        edges = self.edges.keys()
        for e in edges:
            prefix = commonprefix(e, label)
            if prefix == b'':
                # no common prefix
                continue
            if e == label:
                # we're done, set data and probability (or raise exception if double insert)
                if self.edges[e].probability is not None:
                    raise Exception('element already in trie with probability')
                self.edges[e].token = token
                self.edges[e].probability = probability
                self.lookup[token] = self.edges[e]
                return self.edges[e]
            if len(prefix) == len(e):
                # bubble down: append token as child in e
                return self.edges[e].insert(label[len(prefix):], token, probability)
            else:
                # split edge: create new prefix edge, add both e[len(prefix):] and token[len(prefix):] as children
                t = TokenTrie(prefix, parent=self)
                curr = self.edges[e]
                del self.edges[e]
                curr.label = curr.label[len(prefix):]
                curr.parent = t
                from collections import OrderedDict
                # rehang curr
                t.edges[e[len(prefix):]] = curr
                ret: TokenTrie
                if label[len(prefix):]:
                    ret = TokenTrie(label[len(prefix):], token=token, parent=t, probability=probability)
                    t.edges[label[len(prefix):]] = ret
                else:
                    t.token = token
                    t.probability = probability
                    ret = t
                self.edges[prefix] = t
                self.lookup[token] = ret
                return ret
        self.edges[label] = TokenTrie(label, token=token, probability=probability, parent=self)
        self.lookup[token] = self.edges[label]
        return self.edges[label]

    def tokens(self) -> List[int]:
        return ([self.token] if self.token is not None else []) + list(flat_map(lambda x: x.tokens(), self.edges.values()))

    def probabilities(self) -> list[int]:
        return ([self.probability] if self.probability is not None else []) \
            + list(flat_map(lambda x: x.probabilities(), self.edges.values()))

    def distribution(self) -> List[Tuple[int, List[int], int]]:
        if self.token is None:
            # the current node is a split node. Bubble down
            distr = []
            for t in self.edges.values():
                distr += t.distribution()
            return distr
        if not self.edges:
            if self.probability is None:
                return []
            # the current node is a leaf
            return [(self.token, [self.token], self.probability)]
        probs = sum(self.probabilities())
        if probs == 0:
            # empty subtree
            return []
        # the current node has a probability assigned and child nodes -> not uniquely decodable
        return [(self.token, self.tokens(), sum(self.probabilities()))]

    def visualize(self, level=0, max_depth=None) -> str:
        # todo graphviz?
        label = f"{self.label}"
        if self.probability is not None:
            label += f' ({self.token}, {self.probability})'
        indent = '    ' * level
        tr = f'{label}'
        if max_depth and level >= max_depth and self.depth() > 0:
            tr += '\n' + indent + f'├── ... ({self.depth()} levels)'
            return tr
        for e in self.edges.keys():
            tr += '\n' + indent + '├── ' + self.edges[e].visualize(level + 1, max_depth=max_depth)
        return tr

    def depth(self):
        if not self.edges:
            return 0
        return 1 + max(map(lambda t: t.depth(), self.edges.values()))

    def subtree(self, token: int) -> Optional['TokenTrie']:
        return self.lookup[token] if token in self.lookup else None

    def __eq__(self, other):
        return self.label == other.label \
            and self.token == other.token \
            and self.edges == other.edges \
            and self.probability == other.probability \
            and self.parent == other.parent

    def __str__(self):
        return self.visualize()


def commonprefix(x: bytes, y: bytes, carry: List[int] = None) -> bytes:
    if carry is None:
        carry = []
    if not x or not y:
        return bytes(carry)
    if len(x) == 0 or len(y) == 0:
        return bytes(carry)
    if x[0] == y[0]:
        return commonprefix(x[1:], y[1:], carry + [x[0]])
    return bytes(carry)


def flat_map(f, xs):
    return [y for ys in xs for y in f(ys)]


def test_webex_example():
    trie = TokenTrie()
    trie.insert(b'Alice', 0, 0.3)
    trie.insert(b'found', 1, 0.1)
    trie.insert(b'an', 2, 0.2)
    trie.insert(b'ant', 3, 0.1)
    trie.insert(b'at', 4, 0.05)
    trie.insert(b'the', 5, 0.05)
    trie.insert(b'tree', 6, 0.2)
    reprs, tokens, probs = zip(*trie.distribution())
    assert reprs == (0, 1, 2, 4, 5, 6)
    assert tokens[2] == [2, 3]
    for i in range(0, len(reprs)):
        if i == 2:
            continue
        assert tokens[i][0] == reprs[i] and len(tokens[i]) == 1


def test_resample():
    trie = TokenTrie()
    trie.insert(b'Alice', token=1, probability=3)
    trie.insert(b'an', token=2, probability=3)
    trie.insert(b'ant', token=3, probability=4)
    reprs, tokens, probs = zip(*trie.distribution())
    assert reprs[1] == 2
    assert tokens[1] == [2, 3]
    assert probs[1] == 7
    assert tokens[1] == trie.subtree(reprs[1]).tokens()
    assert probs[1] == sum(trie.subtree(reprs[1]).probabilities())


def test_resample_deep():
    trie = TokenTrie()
    trie.insert(b'a', 0, 1)
    trie.insert(b'alice', 1, 3)
    trie.insert(b'an', 2, 3)
    trie.insert(b'ant', 3, 4)
    trie.insert(b'bob', 4, 5)
    reprs, tokens, probs = zip(*trie.distribution())
    assert reprs[0] == 0
    assert tokens[0] == [0, 1, 2, 3]
    assert probs[0] == 11


def test_resample_multi_split():
    trie = TokenTrie()
    trie.insert(b'alice', 0, 3)
    trie.insert(b'an', 1, 3)
    trie.insert(b'albert', 2, 4)
    trie.insert(b'ant', 3, 5)
    trie.insert(b'bob', 4, 7)
    trie.insert(b'a', 5, 1)
    reprs, tokens, probs = zip(*trie.distribution())
    assert reprs[0] == 5
    assert tokens[0] == [5, 1, 3, 0, 2]
    assert probs[0] == 16
    st = trie.subtree(reprs[0])
    assert st.tokens() == [5, 1, 3, 0, 2]
    assert st.probabilities() == [1, 3, 5, 3, 4]
    reprs, tokens, probs = zip(*st.distribution())
    assert tokens[0] == [5, 1, 3, 0, 2]
    assert reprs[0] == 5
    assert probs[0] == 16
    assert len(probs) == 1
    assert len(reprs) == 1
    st = st.subtree(reprs[0])
    assert st.probabilities() == [1, 3, 5, 3, 4]


def test_resample_multi_split_pseudo():
    trie = TokenTrie()
    trie.insert(b'alice', 0, 3)
    trie.insert(b'an', 1, 3)
    trie.insert(b'albert', 2, 4)
    trie.insert(b'ant', 3, 5)
    trie.insert(b'bob', 4, 7)
    reprs, tokens, probs = zip(*trie.distribution())
    assert reprs[0] == 1
    assert tokens[0] == [1, 3]
    assert probs[0] == 8
    st = trie.subtree(reprs[0])
    assert st is not None
    reprs, tokens, probs = zip(*st.distribution())
    assert tokens[0] == [1, 3]
    assert reprs[0] == 1
    assert probs[0] == 8
    assert len(probs) == 1
    assert len(reprs) == 1
    st = st.subtree(reprs[0])
    assert st.probabilities() == [3, 5]


def test_empty_label_split():
    trie = TokenTrie()
    trie.insert(b'ABC', token=234, probability=2)
    trie.insert(b'AB', token=123, probability=3)
    assert trie.edges[b'AB'].token == 123
    assert trie.edges[b'AB'].probability == 3
    assert b'' not in trie.edges[b'AB'].edges
