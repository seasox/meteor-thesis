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

    def prepare_model_inputs(self: GPT2LMHeadModel, inputs: torch.Tensor, num_return_sequences=None, bos_token_id=None, output_attentions=None, output_hidden_states=None, use_cache=None, **model_kwargs):
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
    text = "I am feeling great. I hope you are fine too?"

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

    print_graph(graph)
    paths = all_paths(graph, text)
    print(len(paths))
    #print(paths)
    #print(graph[text])