


def compare_tokens(encode_tokens, decode_tokens):
    i = 0
    j = 0
    num_mismatch = 0
    cum_mismatch_len = 0
    is_dec_longer = False
    curr_mismatch = None
    mismatch_enc_tokens = []
    mismatch_dec_tokens = []
    curr_prefix = None
    mismatches = []
    enclen = len(encode_tokens)
    declen = len(decode_tokens)
    len_offset = abs(declen - enclen)
    while i < enclen or j < declen:
        enc_tok = encode_tokens[i] if i < enclen else None
        dec_tok = decode_tokens[j] if j < declen else None
        if enc_tok != dec_tok:
            if curr_mismatch == None:
                if dec_tok is not None:
                    is_dec_longer = len(dec_tok) > len(enc_tok) if enc_tok is not None else True
                elif enc_tok is not None:
                    is_dec_longer = False
                if is_dec_longer:
                    curr_mismatch = dec_tok
                    mismatch_dec_tokens += [ dec_tok ]
                    curr_prefix = ""
                    j += 1
                else:
                    curr_mismatch = enc_tok
                    mismatch_enc_tokens += [ enc_tok ]
                    curr_prefix = ""
                    i += 1
                num_mismatch += 1
            if is_dec_longer:
                curr_prefix += enc_tok
                mismatch_enc_tokens += [enc_tok]
                i += 1
            else:
                curr_prefix += dec_tok
                mismatch_dec_tokens += [dec_tok]
                j += 1
            cum_mismatch_len += 1
            if curr_prefix.startswith(curr_mismatch):
                if curr_prefix == curr_mismatch:
                    # end of mismatch
                    mismatches = mismatches + [ (mismatch_enc_tokens.copy(), mismatch_dec_tokens.copy()) ]
                    mismatch_enc_tokens = []
                    mismatch_dec_tokens = []
                    curr_mismatch = None
                else:
                    # mismatch overlap
                    is_dec_longer = not is_dec_longer
                    swap = curr_mismatch
                    curr_mismatch = curr_prefix
                    curr_prefix = swap
        else:
            i += 1
            j += 1
    return {
        "num_mismatch": num_mismatch,
        "cum_mismatch_len": cum_mismatch_len,
        "len_offset": len_offset,
        "mismatches": mismatches,
    }


class TestMeteorAnalysis:
    def test_one_mismatch_decoded_longer(self):
        encoded = [ "hello", " world" ]
        decoded = [ "he", "llo", " world" ]

        res = compare_tokens(encoded, decoded)

        assert res["num_mismatch"] == 1
        assert res["len_offset"] == 1
        assert res["cum_mismatch_len"] == 2
        assert res["mismatches"] == [ ([ "hello" ], [ "he", "llo" ]) ]

    def test_one_mismatch_encoded_longer(self):
        encoded = [ "he", "llo", " world" ]
        decoded = [ "hello", " world" ]

        res = compare_tokens(encoded, decoded)

        assert res["num_mismatch"] == 1
        assert res["len_offset"] == 1
        assert res["cum_mismatch_len"] == 2
        assert res["mismatches"] == [ ([ "he", "llo" ], [ "hello" ]) ]

    def test_two_mismatch_decoded_longer(self):
        encoded = [ "hello", " world", " lorem", " ipsum" ]
        decoded = [ "he", "llo", " world", " lor", "em", " ipsum" ]

        res = compare_tokens(encoded, decoded)

        assert res["num_mismatch"] == 2
        assert res["len_offset"] == abs(len(encoded) - len(decoded))
        assert res["cum_mismatch_len"] == 4
        assert res["mismatches"] == [ ([ "hello" ], [ "he", "llo" ]), ([ " lorem" ], [ " lor", "em" ]) ]

    def test_overlapping_mismatch(self):
        # still breaks for overlapping mismatches
        encoded = [ "he", "ll", "o wo", "rld" ]
        decoded = [ "hello", " world" ]

        res = compare_tokens(encoded, decoded)

        assert res["num_mismatch"] == 1
        assert res["len_offset"] == 2
        assert res["cum_mismatch_len"] == 5 # FIXME 4?
        assert res["mismatches"] == [ ([ "he", "ll", "o wo", "rld" ], [ "hello", " world" ]) ]

        return 0

    def test_no_mismatch(self):
        encoded = [ "lorem", " ipsum", " dolor" ]
        decoded = [ "lorem", " ipsum", " dolor" ]

        res = compare_tokens(encoded, decoded)

        assert res["num_mismatch"] == 0
        assert res["len_offset"] == 0
        assert res["cum_mismatch_len"] == 0
        assert res["mismatches"] == []
