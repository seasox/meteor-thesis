import logging
import sys

from util import get_model
from coder import MeteorCoder, MeteorEncryption, DRBG


class PRGEncryption(MeteorEncryption):
    def __init__(self, prg):
        self.prg = prg
        self.encrypt_masks = []
        self.decrypt_masks = []

    def encrypt(self, data, n):
        new_bits = data.copy()
        mask_bits = self.prg.generate_bits(n)
        self.encrypt_masks.append(mask_bits.tobytes())
        for b in range(0, len(new_bits)):
            new_bits[b] = new_bits[b] ^ mask_bits[b]
        return new_bits

    def decrypt(self, data, n):
        new_bits = data.copy()
        mask_bits = self.prg.generate_bits(n)
        self.decrypt_masks.append(mask_bits.tobytes())
        for b in range(0, len(new_bits)):
            new_bits[b] = new_bits[b] ^ mask_bits[b]
        return new_bits


def main():
    model_name = 'gpt2-medium'
    device = 'cpu'

    print('get model')
    logging.basicConfig(level=logging.DEBUG)
    enc, model = get_model(model_name=model_name, device=device)
    print('done')

    coder = MeteorCoder(enc, model, device)

    # Constants for HMAC-DRBG -- MUST CHANGE FOR SECURE IMPLEMENTATION
    sample_seed_prefix = b'sample'
    sample_key = b'0x01' * 64
    sample_nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    enc_prg = DRBG(sample_key, sample_seed_prefix + sample_nonce)
    encryption = PRGEncryption(enc_prg)
    dec_prg = DRBG(sample_key, sample_seed_prefix + sample_nonce)
    decryption = PRGEncryption(dec_prg)
    enc_prg.key = b';\xd0U\x82i^\xb4\xc0\x81\xfb\x07h\xba\xf0\xa6\x04\xea\x8e\x9b\xda+\x04\x8c\xdfV\xe4\xdf\xa4\x15\x83\x11\xe9A\x8c\xed.\x0cd\xbc\xe3\x08$\xdf\x93\xf0k\xd42B\xc9\xcd0Q\xff\x8b\xb9\x8c\xd7.\xac\xcb\xf5n\xf4'
    enc_prg.val = b"\xd0ut\xc6\xfe\xe1[ d\xb8\x10\x9a\xefI\x80\xbe6\xb7A\xfe[\x92s\xc1\x89\x1e\xcb\xc5P\x85\x81\xf8~\xb0{DWi\xb0\x83|3\xcbQ\x9c\xb7\x0bJ\x95\x06\x8aP\x85\xbe\x1ds\x1bJT\xf6'-]\xf9"
    dec_prg.key = b';\xd0U\x82i^\xb4\xc0\x81\xfb\x07h\xba\xf0\xa6\x04\xea\x8e\x9b\xda+\x04\x8c\xdfV\xe4\xdf\xa4\x15\x83\x11\xe9A\x8c\xed.\x0cd\xbc\xe3\x08$\xdf\x93\xf0k\xd42B\xc9\xcd0Q\xff\x8b\xb9\x8c\xd7.\xac\xcb\xf5n\xf4'
    dec_prg.val = b"\xd0ut\xc6\xfe\xe1[ d\xb8\x10\x9a\xefI\x80\xbe6\xb7A\xfe[\x92s\xc1\x89\x1e\xcb\xc5P\x85\x81\xf8~\xb0{DWi\xb0\x83|3\xcbQ\x9c\xb7\x0bJ\x95\x06\x8aP\x85\xbe\x1ds\x1bJT\xf6'-]\xf9"
    chosen_context = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"
    message_text = "Hi! Did anyone follow you last night? Are we still up for tommorow? It was 12 am at the market, right?"
    while True:
        x = coder.encode_message(message_text, chosen_context, encryption)
        y = coder.decode_message(x[0], chosen_context, decryption)
        assert y == message_text


if __name__ == '__main__':
    main()
