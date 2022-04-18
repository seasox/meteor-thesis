import sys

from coder import MeteorCoder, MeteorEncryption, DRBG

chosen_context = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"

message_text = "sample text"

# Use gpt2-medium for 345M param model
# Use gpt2-large for 774M param model
def get_model(seed=1234, model_name='gpt2', device='cuda'):
    import numpy as np
    import torch
    from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
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

    def decode(self, token_ids, **kwargs):
        filtered_tokens = self.convert_ids_to_tokens(token_ids)
        text = self.convert_tokens_to_string(filtered_tokens)
        return text
    GPT2Tokenizer.decode = decode

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, 0)
    GPT2Tokenizer._convert_token_to_id = _convert_token_to_id

    return enc, model


class PRGEncryption(MeteorEncryption):
    def __init__(self, prg):
        self.prg = prg

    def encrypt(self, data, n):
        new_bits = data
        mask_bits = self.prg.generate_bits(n)
        for b in range(0, len(new_bits)):
            new_bits[b] = new_bits[b] ^ mask_bits[b]
        return new_bits

    def decrypt(self, data, n):
        new_bits = data
        mask_bits = self.prg.generate_bits(n)
        for b in range(0, len(new_bits)):
            new_bits[b] = new_bits[b] ^ mask_bits[b]
        return new_bits


def main():
    model_name = 'gpt2-medium'
    device = 'cpu'

    enc, model = get_model(model_name=model_name, device=device)

    coder = MeteorCoder(enc, model, device)


    # Constants for HMAC-DRBG -- MUST CHANGE FOR SECURE IMPLEMENTATION
    sample_seed_prefix = b'sample'
    sample_key = b'0x01'*64
    sample_nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'


    encryption = PRGEncryption(DRBG(sample_key, sample_seed_prefix + sample_nonce))
    decryption = PRGEncryption(DRBG(sample_key, sample_seed_prefix + sample_nonce))

    x = coder.encode_message(message_text, chosen_context, encryption)
    y = coder.decode_message(x[0], chosen_context, decryption)


if __name__ == '__main__':
    main()
    quit()
