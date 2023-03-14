import logging
import os
import time
from optparse import OptionParser
from typing import Optional, List

from coder import MeteorCoder, MeteorStatistics
from util import get_model


def get_random_wiki():
    from wikipedia import wikipedia
    random_article = None
    random_summary = None
    for i in range(0, 100):
        try:
            random_article = wikipedia.random(1)
            random_summary = wikipedia.summary(random_article)
            break
        except Exception as e:
            #logging.exception("retry wikipedia due to exception", exc_info=e)
            random_article, random_summary = None, None
            continue
    if not random_article:
        raise Exception('max retry exceeded')
    return random_article, random_summary


def file_read_bin(fname: str) -> Optional[bytes]:
    if not fname:
        return None
    f = open(fname, 'rb')
    b = f.read()
    f.close()
    return b


def parse_options():
    parser = OptionParser()
    parser.add_option('--message', dest='message', help='message to encode', default='water')
    parser.add_option('--stegotext', dest='stegotext', help='stegotext to decode', default=None)
    parser.add_option('--random-context', action='store_true', dest='random_context', help='Use a random context from wikipedia', default=False)
    parser.add_option('--context', dest='context', help='initial context', default='Give me a good example for a dilemma.\n\n')
    parser.add_option('--key', dest='key', help='key file (defaults to random key)', default=None)
    parser.add_option('--key-out', dest='key_out', help='key out file (defaults to key.bin)', default='key.bin')
    parser.add_option('--nonce', dest='nonce', help='nonce (defaults to random nonce)', default=None)
    parser.add_option('--nonce-out', dest='nonce_out', help='nonce out file (defaults to random nonce.bin)', default='nonce.bin')
    parser.add_option('--repeat', action='store_true', dest='repeat', help='repeat encode/decode (test mode)', default=False)
    parser.add_option('--stats', dest='stats', help='statistics output file', default=None)
    parser.add_option('-n', dest='count', help='number of repetitions (mostly for statistic runs', default=1, type=int)
    (options, args) = parser.parse_args()

    return options, args


def load_stats(fname) -> List[MeteorStatistics]:
    import pickle
    if not os.path.exists(fname):
        return []
    with open(fname, 'rb') as f:
        return pickle.load(f)


def save_stats(fname, stats: List[MeteorStatistics]):
    import pickle
    with open(fname, 'wb+') as f:
        pickle.dump(stats, f)


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    options, args = parse_options()

    logging.info('get model')
    model_name = 'gpt2-medium'
    device = 'cpu'
    enc, model = get_model(model_name=model_name, device=device)
    logging.info('done')

    coder = MeteorCoder(enc, model, device)

    chosen_context = options.context
    message_text = options.message
    repeat = True
    num_rounds = 0
    while repeat:
        if options.random_context:
            chosen_context = get_random_wiki()[1] + '\n\n'
        key = file_read_bin(options.key) or os.urandom(64)
        nonce = file_read_bin(options.nonce) or os.urandom(64)
        start = time.time()
        stegotext = options.stegotext
        if stegotext is None:
            print(f'chosen_context = "{chosen_context}"')
            print(f'key = {key}')
            print(f'nonce = {nonce}')
            stegotext, enc_tokens, stats = coder.encode_message(message_text, chosen_context, key, nonce, coding='arithmetic')
            assert enc.decode(enc.encode(enc_tokens))[0]==stegotext
            print(f"stegotext = {stegotext.encode('utf-8', errors=enc.errors)}.decode('utf-8', errors=enc.errors)")
            print(f'enc_tokens = {enc_tokens}')
            end = time.time()
            logging.info("Encode took {:.02f} s; generated {} bytes of stegotext".format(end - start, len(stegotext)))
            logging.info(f'save key to file {options.key_out}')
            f = open(options.key_out, 'wb')
            f.write(key)
            f.close()
            logging.info(f'save nonce to file {options.nonce_out}')
            f = open(options.nonce_out, 'wb')
            f.write(nonce)
            f.close()
            if options.stats:
                stats.timing = end - start
                logging.info(stats)
                all_stats = load_stats(options.stats)
                all_stats.append(stats)
                save_stats(options.stats, all_stats)
        else:
            enc_tokens = None
        if options.stegotext is not None or options.repeat:
            start = time.time()
            y = coder.decode_message(stegotext, chosen_context, key, nonce, coding='arithmetic', enc_tokens=enc_tokens)
            end = time.time()
            logging.info("Decode took {:.02f} s".format(end - start))
            assert y[:len(message_text)] == message_text[:len(y)], f'{y} != {message_text}'
            if y != message_text:
                logging.warning(f'WARNING: recovered message has additional data or not embedded completely: y={y}; message_text={message_text}')
        options.count -= 1
        repeat = options.repeat or options.count
        num_rounds += 1
        logging.info(f'total rounds: {num_rounds}')


if __name__ == '__main__':
    main()
