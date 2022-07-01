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

    tokenize_cache = {}

    def tokenize_candidates(self, parent, start=True): # parent: Union[str, Node]
        from anytree import Node
        import sys
        if isinstance(parent, str):
            # apply byte-encoding to text
            # this works for python 3 only
            if sys.version_info[0] == 2:
                text = ''.join(self.byte_encoder[ord(b)] for b in parent)
            else:
                text = ''.join([self.byte_encoder[b] for b in parent.encode('utf-8')])
            parent = Node(None, parent=None, text=text)
        if parent.text == '':
            return parent
        print("text: " + parent.text)
        for token, id in self.encoder.items():
            if parent.__dict__['text'].startswith(token):
                self.tokenize_candidates(Node(token, parent=parent, text=parent.__dict__['text'][len(token):]))
        return parent
    GPT2Tokenizer.tokenize_candidates = tokenize_candidates

    return enc, model


if __name__ == '__main__':
    text = """
    "Steganography", "breaking the pen", "to embed text in a message", "hot spread", "until its cover is removed", or whatever, has a long history among computer scientists celebrated by publications like Hunt Eilers' Frauenkumme[4] and that of Dr. Ralf Corroon in particular. Or the respected Peter Iversen[5] who has been working on Steganography and NetWrappers since 1989[6] and you could say that it's been "tovert[ing] for years".

    Armed Upon Me![7] was an excellent article about Parrot Steganography system and its derivatives that has good graphics what to be original and features between the quoted text and the "hidden" image-fiche and snapshot files specially written for peer review, but also Steve Fox who has released allowed full text of both elements[8]

    The term "Steganography"(stype "e") is derived from the German vs in Roman Meaning with an AI"""
    text = 'hello world'
    from anytree import RenderTree

    enc, model = get_model(device='cpu')
    tree = enc.tokenize_candidates(text)
    print(RenderTree(tree))
