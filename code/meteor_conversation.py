import logging
import os
import sys

from util import get_model
from coder import MeteorCoder


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
    chosen_context = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"
    message_text = "Hi! Did anyone follow you last night? Are we still up for tommorow? It was 12 am at the market, right?"
    while True:
        # key = os.urandom(64)
        # nonce = os.urandom(64)
        import time
        start = time.time()
        x = coder.encode_message(message_text, chosen_context, key, nonce)
        end = time.time()
        print("Encode took {:.02f} s".format(end-start))
        start = time.time()
        y = coder.decode_message(x[0], chosen_context, key, nonce)
        end = time.time()
        print("Decode took {:.02f} s".format(end-start))
        assert y[0] == message_text


if __name__ == '__main__':
    main()
