import logging
import os
import pickle
import signal
import sys

from meteor_analysis import compare_tokens
from util import get_model
from coder import MeteorCoder


def write_mismatches(mismatches):
    print("write mismatches...")
    f = open("mismatches.bin", "wb+")
    pickle.dump(mismatches, f)
    f.close()
    print("done")


def load_mismatches():
    print("load mismatches...")
    f = open("mismatches.bin", "rb")
    o = pickle.load(f)
    f.close()
    print("done")
    return o


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
    comparisons = load_mismatches()
    failures = comparisons[len(comparisons)-1]['failures']
    count = comparisons[len(comparisons)-1]['count']
    num_encoded_tokens = comparisons[len(comparisons)-1]['encoded_tokens']
    num_decoded_tokens = comparisons[len(comparisons)-1]['decoded_tokens']
    num_mismatch = comparisons[len(comparisons)-1]['num_mismatch']
    while True:
        key = os.urandom(64)
        nonce = os.urandom(64)
        import time
        start = time.time()
        x = coder.encode_message(message_text, chosen_context, key, nonce)
        end = time.time()
        print("Encode took {:.02f} s".format(end-start))
        start = time.time()
        #y = coder.decode_message(x[0], chosen_context, key, nonce)
        dec_toks = enc.tokenize(x[0])
        end = time.time()
        print("Decode took {:.02f} s".format(end-start))
        count += 1
        num_encoded_tokens += len(x[1])
        num_decoded_tokens += len(dec_toks)
        # decode failed
        failures += 1 if dec_toks != x[1] else 0
        # log # of mismatching tokens
        comparison = compare_tokens(x[1], dec_toks)
        num_mismatch += comparison["num_mismatch"]
        comparison.update({
            "encoded_tokens": num_encoded_tokens,
            "decoded_tokens": num_decoded_tokens,
            "failures": failures,
            "count": count,
            "num_mismatch": num_mismatch,
        })
        comparisons += [comparison]
        write_mismatches(comparisons)
        print(comparison)
        print("encode: ", x[1])
        print("decode: ", dec_toks)

        print("{}/{}={}".format(failures, count, failures/count))
        print("encoded tokens = ", num_encoded_tokens)
        print("decoded tokens = ", num_decoded_tokens)
        print("mismatches = ", num_mismatch)
        print("encoded tokens per mismatch = ", num_encoded_tokens/num_mismatch if num_mismatch > 0 else 0)
        print("decoded tokens per mismatch = ", num_decoded_tokens/num_mismatch if num_mismatch > 0 else 0)


if __name__ == '__main__':
    main()