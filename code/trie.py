class TokenTrie:
    def __init__(self, label=None, probability: float = None, token=None, edges=None, parent=None):
        from collections import OrderedDict
        if edges is None:
            edges = OrderedDict()
        self.label = label
        self.token = token
        self.probability = probability
        self.edges = OrderedDict(edges)  # label: node
        self.parent = parent

    def insert(self, label, probability=None, token=None) -> bool:
        if token is None:
            token = label
        edges = self.edges.keys()
        for e in edges:
            import os
            prefix = os.path.commonprefix([e, label])
            if prefix == '':
                # no common prefix
                continue
            if e == label:
                # we're done, set data and probability (or raise exception if double insert)
                if self.edges[e].probability is not None:
                    raise Exception('element already in trie with probability')
                self.edges[e].token = token
                self.edges[e].probability = probability
                return True
            if len(prefix) == len(e):
                # bubble down: append token as child in e
                return self.edges[e].insert(label[len(prefix):], probability, token)
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
                if label[len(prefix):]:
                    t.edges[label[len(prefix):]] = TokenTrie(label[len(prefix):], token=token, parent=t, probability=probability)
                else:
                    t.token = token
                    t.probability = probability
                self.edges[prefix] = t
                return True
        self.edges[label] = TokenTrie(label, token=token, probability=probability, parent=self)
        return True

    def tokens(self):
        return ([self.token] if self.token is not None else []) + list(flat_map(lambda x: x.tokens(), self.edges.values()))

    def probabilities(self):
        return ([self.probability] if self.probability is not None else []) + list(flat_map(lambda x: x.probabilities(), self.edges.values()))

    def distribution(self):
        if self.token is None:
            # the current node is a split node. Bubble down
            distr = []
            for t in self.edges.values():
                distr += t.distribution()
            return distr
        elif not self.edges:
            # the current node is a leaf
            distr = [(self.token, [self.token], self.probability)]
        else:
            # the current node has a probability assigned and child nodes -> not uniquely decodable
            distr = [(self.token,
                self.tokens(), sum(self.probabilities())
            )]
        return distr

    def visualize(self, level=0, max_depth=None) -> str:
        # todo graphviz?
        label = self.label
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

    def subtree(self, suffix):
        if suffix == '' or self.token == suffix:
            return self
        for e in self.edges.keys():
            if e == suffix or (suffix.startswith(e) and e != ''):
                return self.edges[e].subtree(suffix[len(e):])
        return None

    def __str__(self):
        return self.visualize()


def flat_map(f, xs):
    return [y for ys in xs for y in f(ys)]


def test_webex_example():
    trie = TokenTrie()
    trie.insert('Alice', 0.3)
    trie.insert('found', 0.1)
    trie.insert('an', 0.2)
    trie.insert('ant', 0.1)
    trie.insert('at', 0.05)
    trie.insert('the', 0.05)
    trie.insert('tree', 0.2)
    reprs, tokens, probs = zip(*trie.distribution())
    assert reprs == ('Alice', 'found', 'an', 'at', 'the', 'tree')
    assert tokens[2] == ['an', 'ant']
    for i in range(0, len(reprs)):
        if i == 2:
            continue
        assert tokens[i][0] == reprs[i] and len(tokens[i]) == 1


def test_resample():
    trie = TokenTrie()
    trie.insert('Alice', 3)
    trie.insert('an', 3)
    trie.insert('ant', 4)
    reprs, tokens, probs = zip(*trie.distribution())
    assert reprs[1] == 'an'
    assert tokens[1] == ['an', 'ant']
    assert probs[1] == 7
    assert tokens[1] == trie.subtree(reprs[1]).tokens()
    assert probs[1] == sum(trie.subtree(reprs[1]).probabilities())


def test_resample_deep():
    trie = TokenTrie()
    trie.insert('a', 1)
    trie.insert('alice', 3)
    trie.insert('an', 3)
    trie.insert('ant', 4)
    trie.insert('bob', 5)
    reprs, tokens, probs = zip(*trie.distribution())
    assert reprs[0] == 'a'
    assert tokens[0] == ['a', 'alice', 'an', 'ant']
    assert probs[0] == 11


def test_resample_multi_split():
    trie = TokenTrie()
    trie.insert('alice', 3)
    trie.insert('an', 3)
    trie.insert('albert', 4)
    trie.insert('ant', 5)
    trie.insert('bob', 7)
    trie.insert('a', 1)
    reprs, tokens, probs = zip(*trie.distribution())
    assert reprs[0] == 'a'
    assert tokens[0] == ['a', 'an', 'ant', 'alice', 'albert']
    assert probs[0] == 16
    st = trie.subtree(reprs[0])
    assert st.tokens() == ['a', 'an', 'ant', 'alice', 'albert']
    assert st.probabilities() == [1, 3, 5, 3, 4]
    reprs, tokens, probs = zip(*st.distribution())
    assert tokens[0] == ['a', 'an', 'ant', 'alice', 'albert']
    assert reprs[0] == 'a'
    assert probs[0] == 16
    assert len(probs) == 1
    assert len(reprs) == 1
    st = st.subtree(reprs[0])
    assert st.probabilities() == [1, 3, 5, 3, 4]


def test_resample_multi_split_pseudo():
    trie = TokenTrie()
    trie.insert('alice', 3)
    trie.insert('an', 3)
    trie.insert('albert', 4)
    trie.insert('ant', 5)
    trie.insert('bob', 7)
    reprs, tokens, probs = zip(*trie.distribution())
    assert reprs[0] == 'an'
    assert tokens[0] == ['an', 'ant']
    assert probs[0] == 8
    print(trie)
    st = trie.subtree(reprs[0])
    assert st is not None
    reprs, tokens, probs = zip(*st.distribution())
    assert tokens[0] == ['an', 'ant']
    assert reprs[0] == 'an'
    assert probs[0] == 8
    assert len(probs) == 1
    assert len(reprs) == 1
    st = st.subtree(reprs[0])
    assert st.probabilities() == [3, 5]


def test_empty_label_split():
    trie = TokenTrie()
    trie.insert('ABC', 2, token=234)
    trie.insert('AB', 3, token=123)
    assert trie.edges['AB'].token == 123
    assert trie.edges['AB'].probability == 3
    assert '' not in trie.edges['AB'].edges
