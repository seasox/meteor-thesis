def ld():
	f = open("mismatches.bin", "rb")
	import pickle
	o = pickle.load(f)
	f.close()
	return o


if __name__ == '__main__':
	data = ld()
	encoded_tokens = list(map(lambda x: len(x.encoded_tokens), data))
	mismatch_stat = list(map(lambda x: len(x.encoded_tokens)/len(x.mismatches) if len(x.mismatches) > 0 else 0, data))
	bits_per_word = list(map(lambda x: 1/x.stats['wordsbit'], data))
	kl = list(map(lambda x: 1/x.stats['kl'], data))
	ppl = list(map(lambda x: 1/x.stats['ppl'], data))
	avg_bits_per_word = []
	for i in range(len(data)):
		prev = avg_bits_per_word[i-1] if i > 0 else 0
		avg_bits_per_word += [(prev * i + 1/data[i].stats['wordsbit']) / (i+1)]

	flat_map = lambda f, xs: [y for ys in xs for y in f(ys)]
	mismatches = list(flat_map(lambda x: x.mismatches, data))

	print(mismatches)


	#mismatches = list(map(lambda x: x['mismatches'], data))
	#print(mismatches)
	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(3)
	axs[0].plot(bits_per_word, label="bits per word")
	axs[0].plot(avg_bits_per_word, label="avg bits per word")
	axs[1].plot(mismatch_stat, label="mismatch rate")
	#axs[2].plot(kl, label="KL")
	#axs[1].ylabel('mismatches per encoded tokens')

	plt.show()
