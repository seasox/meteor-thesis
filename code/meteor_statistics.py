import logging
import os
import pickle

from meteor_analysis import compare_tokens
from util import get_model
from coder import MeteorCoder


def write_mismatches(mismatches):
    print("write mismatches...")
    f = open("meteor_statistics.pickle", "wb+")
    pickle.dump(mismatches, f)
    f.close()
    print("done")


def load_mismatches():
    print("load mismatches...")
    from os.path import exists
    if exists("meteor_statistics.pickle"):
        f = open("meteor_statistics.pickle", "rb")
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
    chosen_context = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"
    #message_text = "Hi! Did anyone follow you last night? Are we still up for tommorow? It was 12 am at the market, right?"
    import coder as codr
    context_tokens = codr.encode_context(chosen_context, enc)
    from os.path import exists
    message_text = open('hamlet_act1.txt', 'r').read()
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

    comparisons = load_mismatches()
    i = 0
    hamlet_len = len(message_text)
    step_size = 128
    while i < hamlet_len:
        coding = 'utf-8'
        key = os.urandom(64)
        nonce = os.urandom(64)
        import time
        start = time.time()
        text, enc_toks, stats = coder.encode_message(message_text[i:i+step_size], chosen_context, key, nonce, coding=coding)
        end = time.time()
        print("Encode took {:.02f} s".format(end-start))
        start = time.time()
        #y = coder.decode_message(x[0], chosen_context, key, nonce)
        dec_toks = enc.tokenize(text)
        end = time.time()
        print("Decode took {:.02f} s".format(end-start))
        num_encoded_tokens = len(enc_toks)
        num_decoded_tokens = len(dec_toks)
        # log comparison statistics
        comparison = compare_tokens(message_text[i:i+step_size].encode('utf-8'), chosen_context, key, nonce, coding, enc_toks, dec_toks, stats)
        comparisons += [comparison]
        num_mismatch = len(comparison.mismatches)
        write_mismatches(comparisons)
        print(comparison)
        print("encode: ", enc_toks)
        print("decode: ", dec_toks)

        print("mismatches = ", num_mismatch)
        print("encoded tokens per mismatch = ", num_encoded_tokens/num_mismatch if num_mismatch > 0 else 0)
        print("decoded tokens per mismatch = ", num_decoded_tokens/num_mismatch if num_mismatch > 0 else 0)
        i += step_size


if __name__ == '__main__':
    main()
