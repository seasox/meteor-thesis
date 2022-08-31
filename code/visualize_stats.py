import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib


def ld(fname):
    f = open(fname, "rb")
    import pickle
    o = pickle.load(f)
    f.close()
    return o


if __name__ == '__main__':
    def flat_map(f, xs):
        return [y for ys in xs for y in f(ys)]


    tikzexport = True

    plt.figure(dpi=1200)
    for step_size in [128, 1024]:
        fname = f'meteor_statistics_{step_size}.pickle'
        data = ld(fname)
        print(f'generating {step_size} bytes stats')
        data_w_coding = list(filter(lambda x: x.coding == 'arithmetic', data))
        if not data_w_coding:
            continue
        # print(data)
        mismatched_codings = list(filter(lambda x: len(x.mismatches) > 0, data_w_coding))
        encoded_tokens = list(map(lambda x: len(x.encoded_tokens), data_w_coding))
        mismatch_count = list(map(lambda x: len(x.mismatches), data_w_coding))
        mismatch_prob = len(list(filter(lambda x: x == 0, mismatch_count))) / len(mismatch_count)
        print(mismatch_prob)
        mismatch_rate = list(filter(lambda x: x != float('inf'), map(lambda x: x[0] / x[1] if x[1] != 0 else 0,
                                                                     zip(encoded_tokens, mismatch_count))))
        avg_mismatch_rate = [sum(encoded_tokens) / sum(mismatch_count) for _ in range(len(encoded_tokens))]
        mismatches = list(flat_map(lambda x: x.mismatches, data_w_coding))
        mismatch_len = list(map(lambda x: len(x[3]), mismatches))
        bits_per_word = list(flat_map(lambda x: x.stats['encoded_bits_in_output'], data_w_coding))
        avg_bits = np.mean(bits_per_word)
        avg_bits_per_word = [avg_bits for i in range(len(bits_per_word))]
        kl = list(map(lambda x: 1 / x.stats['kl'], data_w_coding))
        ppl = list(map(lambda x: 1 / x.stats['ppl'], data_w_coding))

        histbins = np.arange(min(mismatch_count) - 0.5, max(mismatch_count) + 1.5)
        plt.xlim(min(histbins), max(histbins))
        plt.hist(mismatch_count, bins=histbins, label="mismatch count", align='mid')
        plt.tight_layout()
        plt.legend()
        if tikzexport:
            tikzplotlib.save(f'../tex/fig_meteor_stats_mismatch_count_{step_size}.tikz')
        else:
            plt.show()
        plt.clf()

        # plt.annotate('%E' % avg_mismatch_rate[-1], (len(avg_mismatch_rate), avg_mismatch_rate[-1]))
        # plt.plot(avg_mismatch_rate, label="avg mismatch rate")
        # xaxis = plt.gca()
        # xaxis.set_visible(False)
        plt.hist(encoded_tokens, bins=np.arange(min(encoded_tokens) - 0.5, max(encoded_tokens) + 1.5),
                 label='encoded tokens')
        plt.tight_layout()
        plt.legend()
        if tikzexport:
            tikzplotlib.save(f'../tex/fig_meteor_stats_encoded_tokens_{step_size}.tikz')
        else:
            plt.show()
        plt.clf()

        plt.hist(mismatch_len, bins=np.arange(min(mismatch_len) - 0.5, max(mismatch_len) + 1.5),
                 label='mismatch lengths')
        plt.tight_layout()
        plt.legend()
        if tikzexport:
            tikzplotlib.save(f'../tex/fig_meteor_stats_mismatch_length_{step_size}.tikz')
        else:
            plt.show()
        plt.clf()
