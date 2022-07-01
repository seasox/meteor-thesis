from functools import reduce

import numpy as np


def ld():
	f = open("meteor_statistics.pickle", "rb")
	import pickle
	o = pickle.load(f)
	f.close()
	return o


if __name__ == '__main__':
	flat_map = lambda f, xs: [y for ys in xs for y in f(ys)]
	data = ld()
	encoded_tokens = list(map(lambda x: len(x.encoded_tokens), data))
	mismatch_count = list(map(lambda x: len(x.mismatches), data))
	mismatch_rate = list(map(lambda x: x[0]/x[1] if x[1] != 0 else 0, zip(mismatch_count, encoded_tokens)))
	avg_mismatch_rate = [sum(mismatch_count) / sum(encoded_tokens) for _ in range(len(encoded_tokens))]
	#avg_mismatch_rate = [np.mean(mismatch_stat[:i]) if i > 0 else 0 for i in range(len(mismatch_stat))]
	bits_per_word = list(flat_map(lambda x: x.stats['encoded_bits_in_output'][41:], data))
	avg_bits_per_word = [np.mean(bits_per_word[:i]) if i > 0 else 0 for i in range(len(bits_per_word))]
	kl = list(map(lambda x: 1/x.stats['kl'], data))
	ppl = list(map(lambda x: 1/x.stats['ppl'], data))


	#mismatches = list(map(lambda x: x['mismatches'], data))
	#print(mismatches)
	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(3)
	axs[0].set_title("bits per word")
	axs[0].plot(bits_per_word)
	axs[0].plot(avg_bits_per_word, label="avg bits per word")
	axs[1].set_title("mismatched decodings")
	axs[1].plot(mismatch_count)
	#axs[1].plot(avg_mismatch_rate, label="avg mismatch rate")
	axs[2].set_title("mismatch rate (encoded tokens per mismatch)")
	axs[2].plot(mismatch_rate, label="mismatch rate (encoded tokens per mismatch)")
	axs[2].plot(avg_mismatch_rate, label="avg mismatch rate")
	fig.tight_layout()
	plt.show()
	#import tikzplotlib
	#tikzplotlib.save("plot.tex")
