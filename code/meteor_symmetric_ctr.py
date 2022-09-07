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

    chosen_context = "Hello my friend. I hope you are great. How was your weekend?"
    message_text = "Hello world"
    import os
    key = os.urandom(32)
    while True:
        nonce = os.urandom(64)
        import time
        start = time.time()
        x, enc_tokens, stats = coder.encode_message(message_text, chosen_context, key, nonce, coding='arithmetic',
                                                    randomized=True)
        end = time.time()
        print("Encode took {:.02f} s; generated {} bytes of stegotext".format(end - start, len(x)))
        print("=" * 10 + " stegotext " + "=" * 10)
        print(x)
        print("=" * 30)
        start = time.time()
        y, dec_tokens = coder.decode_message(x, chosen_context, key, nonce, coding='arithmetic', randomized=True)
        end = time.time()
        print("Decode took {:.02f} s".format(end - start))
        print(y)


if __name__ == '__main__':
    main()
