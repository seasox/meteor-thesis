import sys

from util import get_model
from coder import MeteorCoder, MeteorEncryption, DRBG


class PRGEncryption(MeteorEncryption):
    def __init__(self, prg):
        self.prg = prg

    def encrypt(self, data, n):
        new_bits = data.copy()
        mask_bits = self.prg.generate_bits(n)
        for b in range(0, len(new_bits)):
            new_bits[b] = new_bits[b] ^ mask_bits[b]
        return new_bits

    def decrypt(self, data, n):
        new_bits = data.copy()
        mask_bits = self.prg.generate_bits(n)
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
    encryption = PRGEncryption(DRBG(sample_key, sample_seed_prefix + sample_nonce))
    decryption = PRGEncryption(DRBG(sample_key, sample_seed_prefix + sample_nonce))
    chosen_context = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"
    message_text = "Hi! Did anyone follow you last night? Are we still up for tommorow? It was 12 am at the market, right?"
    while True:
        print("DRBG states: %d,%d" % (encryption.prg.key, encryption.prg.val), encryption.prg)
        x = coder.encode_message(message_text, chosen_context, encryption)
        y = coder.decode_message(x[0], chosen_context, decryption)
        assert y == message_text


if __name__ == '__main__':
    main()
