import numpy as np


def ld(fname):
    f = open(fname, "rb")
    import pickle
    o = pickle.load(f)
    f.close()
    return o


if __name__ == '__main__':
    flat_map = lambda f, xs: [y for ys in xs for y in f(ys)]
    fname = 'meteor_statistics.pickle'
    data = ld(fname)

    import matplotlib.pyplot as plt
    import matplotlib.ticker as plticker

    plt.figure(dpi=1200)
    for coding in ['arithmetic', 'utf-8']:
        print(f'generating {coding} stats')
        data_w_coding = list(filter(lambda x: x.coding == coding, data))
        # print(data)
        mismatched_codings = list(filter(lambda x: len(x.mismatches) > 0, data_w_coding))
        encoded_tokens = list(map(lambda x: len(x.encoded_tokens), data_w_coding))
        mismatch_count = list(map(lambda x: len(x.mismatches), data_w_coding))
        mismatch_rate = list(map(lambda x: x[0] / x[1] if x[1] != 0 else x[0], zip(encoded_tokens, mismatch_count)))
        avg_mismatch_rate = [sum(encoded_tokens) / sum(mismatch_count) for _ in range(len(encoded_tokens))]
        # avg_mismatch_rate = [np.mean(mismatch_stat[:i]) if i > 0 else 0 for i in range(len(mismatch_stat))]
        bits_per_word = list(flat_map(lambda x: x.stats['encoded_bits_in_output'], data_w_coding))
        avg_bits = np.mean(bits_per_word)
        # avg_bits_per_word = [np.mean(bits_per_word[:i]) if i > 0 else 0 for i in range(len(bits_per_word))]
        avg_bits_per_word = [avg_bits for i in range(len(bits_per_word))]
        kl = list(map(lambda x: 1 / x.stats['kl'], data_w_coding))
        ppl = list(map(lambda x: 1 / x.stats['ppl'], data_w_coding))

        fig, axs = plt.subplots(4)
        axs[0].set_title("bits per word")
        axs[0].hist(bits_per_word, bins=np.arange(max(bits_per_word)) - 0.5, ec="k")
        # axs[0].plot(avg_bits_per_word, label="avg bits per word")
        axs[1].set_title("mismatched decodings")
        loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
        axs[1].yaxis.set_major_locator(loc)
        axs[1].plot(mismatch_count)
        # axs[1].plot(avg_mismatch_rate, label="avg mismatch rate")
        axs[2].set_title("mismatch rate (#encoded/#mismatch)")
        axs[2].plot(mismatch_rate, label="mismatch rate (#mismatch/#encoded)")
        axs[2].annotate('%E' % avg_mismatch_rate[-1], (len(avg_mismatch_rate), avg_mismatch_rate[-1]))
        axs[2].plot(avg_mismatch_rate, label="avg mismatch rate")
        axs[3].set_title('encoded tokens')
        axs[3].plot(encoded_tokens, label='encoded tokens')
        fig.suptitle(f'{coding} (total: {sum(encoded_tokens)} tokens)')
        fig.tight_layout()
        # plt.legend()
        plt.show()
# plt.savefig('plot.eps', format='eps')
# import tikzplotlib
# tikzplotlib.save("plot.tex")
