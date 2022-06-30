# Use gpt2-medium for 345M param model
# Use gpt2-large for 774M param model
from typing import Tuple, List


def get_model(seed=1234, model_name='gpt2', device='cuda'):
    import numpy as np
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    enc = GPT2Tokenizer.from_pretrained(model_name)
    enc.unk_token = None
    enc.bos_token = None
    enc.eos_token = None

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # model.double()  # want to avoid using this

    def decode(self, token_ids, **kwargs) -> Tuple[str, List[str]]:
        filtered_tokens = self.convert_ids_to_tokens(token_ids)
        text = self.convert_tokens_to_string(filtered_tokens)
        return text, filtered_tokens

    GPT2Tokenizer.decode = decode

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, 0)

    GPT2Tokenizer._convert_token_to_id = _convert_token_to_id

    return enc, model