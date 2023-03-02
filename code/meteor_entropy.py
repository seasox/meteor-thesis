import concurrent.futures
import logging
import multiprocessing.pool
import os
import pickle
import random

from coder import MeteorCoder
from meteor_analysis import compare_tokens
from util import get_model


def write_token_stats(step_size, mismatches):
    print("write mismatches...")
    f = open(f"meteor_statistics_{step_size}.pickle", "wb+")
    pickle.dump(mismatches, f)
    f.close()
    print("done")


def load_mismatches(step_size):
    print("load mismatches...")
    from os.path import exists
    if exists(f"meteor_statistics_{step_size}.pickle"):
        f = open(f"meteor_statistics_{step_size}.pickle", "rb")
        o = pickle.load(f)
        f.close()
        return o
    print("done")
    return []


def main():
    model_name = 'gpt2-medium'
    device = 'cpu'

    print('get model')
    logging.basicConfig(level=logging.DEBUG)
    enc, model = get_model(model_name=model_name, device=device)
    print('done')

    coder = MeteorCoder(enc, model, device)

    # Constants for HMAC-DRBG -- MUST CHANGE FOR SECURE IMPLEMENTATION
    #chosen_context = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"
    # message_text = "Hi! Did anyone follow you last night? Are we still up for tommorow? It was 12 am at the market, right?"
    # context_tokens = codr.encode_context(chosen_context, enc)
    message_text = open('hamlet.txt', 'r').read()
    """if True:
        message_text = open('hamlet_act1.txt', 'r').read()
        message = message_text.encode('utf-8')
    elif not exists('hamlet.pickle'):
        message_text = open('hamlet_act1.txt', 'r').read()
        # encode arithmetic beforehand (takes a long time)
        message = codr.decode_arithmetic(model, enc, message_text, [enc.encoder['<|endoftext|>']], precision=40, topk=60000, device=device)
        hamlet_f = open('hamlet.pickle', 'wb+')
        pickle.dump(message, hamlet_f)
        hamlet_f.close()
    else:
        hamlet_f = open('hamlet.pickle', 'rb')
        message = pickle.load(hamlet_f)
        hamlet_f.close()"""

    hamlet_len = len(message_text)
    def run_thread(i, update_comparisons):
        key = os.urandom(64)
        nonce = os.urandom(64)
        import time
        start = time.time()
        context_start_idx = random.randrange(0, hamlet_len - 128)
        context_str = message_text[context_start_idx: context_start_idx + 128] + '\n\n'
        text, enc_toks, stats = coder.encode_message(message_text[i:i + step_size], context_str, key, nonce,
                                                     coding='arithmetic')
        end = time.time()
        print("Encode took {:.02f} s".format(end - start))
        start = time.time()
        # y = coder.decode_message(x[0], chosen_context, key, nonce)
        dec_toks = enc.tokenize(text)
        end = time.time()
        print("Decode took {:.02f} s".format(end - start))
        num_encoded_tokens = len(enc_toks)
        num_decoded_tokens = len(dec_toks)
        # log comparison statistics
        comparison = compare_tokens(message_text[i:i + step_size].encode('utf-8'), context_str, key, nonce,
                                    'arithmetic',
                                    enc_toks, dec_toks, stats)
        update_comparisons(enc_toks)
        num_mismatch = len(comparison.mismatches)
        print(comparison)
        print("encode: ", enc_toks)
        print("decode: ", dec_toks)

        #i += step_size
        #print("mismatches = ", num_mismatch)
        #print("encoded tokens per mismatch = ", num_encoded_tokens / num_mismatch if num_mismatch > 0 else 0)
        #print("decoded tokens per mismatch = ", num_decoded_tokens / num_mismatch if num_mismatch > 0 else 0)
        #print("total progress = ", i / hamlet_len)
        return
    #threads = []
    import threading
    token_stats = {}
    for step_size in [128]:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=100)
        comparisons_lock = threading.Lock()
        token_stats = {}
        def update_comparisons(tokens):
            print('update_comparisons')
            print(tokens)
            for t in tokens:
                comparisons_lock.acquire()
                if t not in token_stats:
                    token_stats[t] = 1
                else:
                    token_stats[t] += 1
                comparisons_lock.release()
            print(token_stats)
            print('done')
        i = 0
        while i < hamlet_len:
            print(f'spawning {step_size}:{i}')
            pool.submit(run_thread, i, update_comparisons)
            i += step_size
        pool.shutdown()
        print(f'done: {step_size}')
    print('bye')
    print('final result: ')
    print(token_stats)



if __name__ == '__main__':
    main()