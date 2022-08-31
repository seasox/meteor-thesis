import logging

from coder import MeteorCoder
from util import get_model


def main():
    model_name = 'gpt2-medium'
    device = 'cpu'

    print('get model')
    logging.basicConfig(level=logging.DEBUG)
    enc, model = get_model(model_name=model_name, device=device)
    print('done')

    coder = MeteorCoder(enc, model, device)

    chosen_context = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"
    message_text = "Hello world"
    while True:
        import os
        key = os.urandom(64)
        nonce = os.urandom(64)
        import time
        start = time.time()
        x, enc_tokens, stats = coder.encode_message(message_text, chosen_context, key, nonce, coding='arithmetic')
        end = time.time()
        print("Encode took {:.02f} s".format(end - start))
        start = time.time()
        y, dec_tokens = coder.decode_message(x, chosen_context, key, nonce, coding='arithmetic')
        end = time.time()
        print("Decode took {:.02f} s".format(end - start))
        print(y)
        if y[0] != message_text[0]:
            print(key)
            print(enc_tokens)
            print(dec_tokens)
            assert False


if __name__ == '__main__':
    main()
