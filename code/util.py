# Use gpt2-medium for 345M param model
# Use gpt2-large for 774M param model
import sys
from typing import Tuple, List, Optional

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def get_model(seed=1234, model_name='gpt2', device='cuda') -> (GPT2Tokenizer, GPT2LMHeadModel):
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    enc = GPT2Tokenizer.from_pretrained(model_name, errors='surrogateescape')
    # enc.unk_token = None
    # enc.bos_token = None
    # enc.eos_token = None

    # check no token has inner space (which holds for subword tokenizers)
    assert len(list(filter(lambda x: x.find('Ġ') > 0, enc.encoder.keys()))) == 0

    model = GPT2LMHeadModel.from_pretrained(model_name)
    # model.resize_token_embeddings(len(enc))
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

    def prepare_model_inputs(self: GPT2LMHeadModel, inputs: torch.Tensor, num_return_sequences=None, bos_token_id=None,
                             output_attentions=None, output_hidden_states=None, use_cache=None, **model_kwargs):
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        # expect decoder-only
        input_ids = inputs_tensor

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache
        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids,
            expand_size=num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        return input_ids, model_kwargs

    GPT2LMHeadModel.prepare_model_inputs = prepare_model_inputs

    def _tokenize(self, text):
        """Tokenize a string.

        this is an edited version of GPT2Tokenizer._encode. The only change is that we include errors=self.errors, mainly to
        allow the use of surrogate escapes using errors='surrogateescape'
        """
        bpe_tokens = []
        import regex as re
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8", errors=self.errors)
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    GPT2Tokenizer._tokenize = _tokenize

    return enc, model


# the performance of this is really bad
def all_paths(graph, root: tuple[str, str, int, int]) -> [[Optional[tuple[str, str, int, int]]]]:
    if root[0] == '':
        return [[None]]
    queue = []
    hops = graph[root[0]]
    queue += hops
    paths = []
    for hop in hops:
        subpaths = all_paths(graph, hop)
        for subpath in subpaths:
            paths += [[root] + subpath]
    return paths


def bucketize_tokens(tokens):
    bins = {}
    ignored_token_ids = [
        220  # 'space' token
    ]
    for token, id in (t for t in tokens if t[1] not in ignored_token_ids):
        bins[token[0]] = bins.setdefault(token[0], []) + [(token, id)]
    return bins


def tokenize_candidates(text, tokens):
    tokenize_edges = {}
    for i in range(len(text)):
        _do_tokenize_candidates(text[-(i + 1):], tokens, tokenize_edges)
    return tokenize_edges


def _do_tokenize_candidates(text, tokens, tokenize_edges):  # parent: Union[str, Node]
    if text in tokenize_edges or text == '':
        return
    # for token, id in self.encoder.items():
    for token, id in tokens[text[0]]:
        if text.startswith(token):
            remainder = text[len(token):]
            if text not in tokenize_edges:
                tokenize_edges[text] = []
            if remainder not in tokenize_edges[text]:
                tokenize_edges[text] += [(remainder, token, id, -len(token))]


if __name__ == '__main__':
    # text = "I am feeling great. I hope you are fine too?"
    text = "circumstances"

    enc, model = get_model(device='cpu')

    # apply byte-encoding to text
    if sys.version_info[0] == 2:
        text = ''.join(enc.byte_encoder[ord(b)] for b in text)
    else:
        text = ''.join([enc.byte_encoder[b] for b in text.encode('utf-8')])
    token_dict = enc.encoder
    graph = tokenize_candidates(text, bucketize_tokens(token_dict.items()))


    # import timeit
    # print(timeit.timeit(lambda: enc.tokenize_candidates(text), number=1))
    # speedup 4.6 -> 2.3 seconds if sorted tokens

    def print_graph(g):
        for node in g:
            print(node, " ---> ", [i for i in g[node]])


    print_graph(graph)
    paths = all_paths(graph, (text, None, None, None))
    print(len(paths))
    #print(paths)
    # print(graph[text])
