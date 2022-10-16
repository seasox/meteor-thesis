import logging
import time

from coder import MeteorCoder
from util import get_model


def main():
    model_name = 'gpt2-medium'
    device = 'cpu'
    logging.basicConfig(level=logging.DEBUG)
    enc, model = get_model(model_name=model_name, device=device)
    coder = MeteorCoder(enc, model, device)

    key = b'\xb9^\x03\xa1\xd0\x1b0O\x11\xdc\xf2\xbc\x84N_\xd3\xcb\xedA%;\x05\x06\x87`\x04 {,*\x10\xe2\xd8\x9c\x1a@\xe950\xbf\xcf\xaa\xeeT\xe6j\xe0H\xd2\xd2\xa56a[\n\x81\xaf\xe7\x92\x888w\xd5\xb6'
    history = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"
    message = "Hello world"

    stegotext, _, _ = coder.encode_message(message, history, key, b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', coding='arithmetic')
    message_recovered, _ = coder.decode_message(stegotext, history, key, b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', coding='arithmetic')
    print('Decode("%s") = "%s" != "%s"' % (stegotext.replace('\n', '\\n'), message_recovered.replace('\n', '\\n'), message))


if __name__ == '__main__':
    main()
