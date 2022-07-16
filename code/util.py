# Use gpt2-medium for 345M param model
# Use gpt2-large for 774M param model
import sys
from typing import Tuple, List, Optional


def get_model(seed=1234, model_name='gpt2', device='cuda'):
    import numpy as np
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    enc = GPT2Tokenizer.from_pretrained(model_name)
    #enc.add_tokens(['<sp1>', '<sp2>'])
    enc.unk_token = None
    enc.bos_token = None
    enc.eos_token = None

    model = GPT2LMHeadModel.from_pretrained(model_name)
    #model.resize_token_embeddings(len(enc))
    model.to(device)
    model.eval()

    # model.double()  # want to avoid using this

    def decode(self, token_ids, **kwargs) -> Tuple[str, List[str]]:
        filtered_tokens = self.convert_ids_to_tokens(token_ids, kwargs)
        text = self.convert_tokens_to_string(filtered_tokens)
        return text, filtered_tokens

    GPT2Tokenizer.decode = decode

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, 0)

    GPT2Tokenizer._convert_token_to_id = _convert_token_to_id

    def bucketize_tokens(tokens):
        bins = {}
        ignored_token_ids = [
            220  # 'space' token
        ]
        for token, id in (t for t in tokens if t[1] not in ignored_token_ids):
            bins[token[0]] = bins.setdefault(token[0], []) + [(token, id)]
        return bins

    def tokenize_candidates(self, text):
        import sys
        tokens = bucketize_tokens(self.encoder.items())
        tokenize_edges = {}
        for i in range(len(text)):
            _do_tokenize_candidates(self, text[-(i+1):], tokens, tokenize_edges)
        return tokenize_edges

    def _do_tokenize_candidates(self, text, tokens, tokenize_edges): # parent: Union[str, Node]
        from anytree import Node
        if text in tokenize_edges or text == '':
            return
        #for token, id in self.encoder.items():
        for token, id in tokens[text[0]]:
            if text.startswith(token):
                remainder = text[len(token):]
                if text not in tokenize_edges:
                    tokenize_edges[text] = []
                if remainder not in tokenize_edges[text]:
                    tokenize_edges[text] += [(remainder, token, id)]
    GPT2Tokenizer.tokenize_candidates = tokenize_candidates

    return enc, model

# the performance of this is really bad
def all_paths(graph, root) -> [[Optional[str]]]:
    if root == '':
        return [[None]]
    queue = []
    hops = graph[root]
    queue += hops
    paths = []
    for hop in hops:
        subpaths = all_paths(graph, hop[0])
        for subpath in subpaths:
            paths += [[root] + subpath]
    return paths


if __name__ == '__main__':
    text = """
"Steganography", "breaking the pen", "to embed text in a message", "hot spread", "until its cover is removed", or whatever, has a long history among computer scientists celebrated by publications like Hunt Eilers' Frauenkumme[4] and that of Dr. Ralf Corroon in particular. Or the respected Peter Iversen[5] who has been working on Steganography and NetWrappers since 1989[6] and you could say that it's been "tovert[ing] for years".

Armed Upon Me![7] was an excellent article about Parrot Steganography system and its derivatives that has good graphics what to be original and features between the quoted text and the "hidden" image-fiche and snapshot files specially written for peer review, but also Steve Fox who has released allowed full text of both elements[8]

The term "Steganography"(stype "e") is derived from the German vs in Roman Meaning with an AI"""
    text = 'hello world'
    from anytree import RenderTree

    enc, model = get_model(device='cpu')

    # apply byte-encoding to text
    if sys.version_info[0] == 2:
        text = ''.join(enc.byte_encoder[ord(b)] for b in text)
    else:
        text = ''.join([enc.byte_encoder[b] for b in text.encode('utf-8')])
    graph = enc.tokenize_candidates(text)

    #import timeit
    #print(timeit.timeit(lambda: enc.tokenize_candidates(text), number=1))
    # speedup 4.6 -> 2.3 seconds if sorted tokens

    def print_graph(g):
        for node in g:
            print(node, " ---> ", [i for i in g[node]])

    from visualize_stats import ld

    data = ld()
    first_mismatch = next(d for d in data if len(d.mismatches) > 0)

    print_graph(graph)
    #paths = all_paths(graph, text)
    #print(paths)
    #print(graph[text])