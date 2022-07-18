import logging
from typing import Dict

from coder import MeteorCoder, encode_context
from util import get_model


def get_cached_model(model_name, device):
    enc, model = get_model(model_name=model_name, device=device)

    cache: Dict = {}

    def query_cached_model(tensor, past):
        if (tensor, past) not in cache:
            cache[(tensor, past)] = model(tensor, past=past)
        else:
            print('cache hit')
        return cache[(tensor, past)]

    cached_model = query_cached_model

    return enc, cached_model


def generate_message():
    return "Hi! Did anyone follow you last night? Are we still up for tommorow? It was 12 am at the market, right?"


def main():
    model_name = 'gpt2-medium'
    device = 'cpu'

    print('get model')
    logging.basicConfig(level=logging.DEBUG)
    enc, model = get_model(model_name=model_name, device=device)
    print('done')

    coder = MeteorCoder(enc, model, device)

    # Constants for HMAC-DRBG -- MUST CHANGE FOR SECURE IMPLEMENTATION
    key = b'0x01' * 64
    nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    chosen_context_str = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"

    chosen_context = encode_context(chosen_context_str, enc)

    while True:
        message_text = generate_message()
        encoded_message = coder.encode_message(message_text, chosen_context_str, key, nonce)
        print(encoded_message[0])


if __name__ == '__main__':
    main()
