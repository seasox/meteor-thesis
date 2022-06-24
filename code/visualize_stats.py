def ld():
	f = open("mismatches.bin", "rb")
	import pickle
	o = pickle.load(f)
	f.close()
	return o


if __name__ == '__main__':
	data = ld()
	encoded_tokens = list(map(lambda x: x['encoded_tokens'], data))
	mismatch_stat = list(map(lambda x: x['encoded_tokens']/x['num_mismatch'] if x['num_mismatch'] > 0 else 0, data))
	bits_per_word = list(map(lambda x: 1/x['stats']['wordsbit'], data))
	avg_bits_per_word = []
	for i in range(len(data)):
		prev = avg_bits_per_word[i-1] if i > 0 else 0
		avg_bits_per_word += [(prev * i + 1/data[i]['stats']['wordsbit']) / (i+1)]

	#mismatches = list(map(lambda x: x['mismatches'], data))
	#print(mismatches)
	import matplotlib.pyplot as plt
	plt.plot(bits_per_word, label = "bits per word")
	plt.plot(avg_bits_per_word, label = "avg bits per word")
	#plt.plot(mismatch_stat, label="mismatch rate")
	#plt.ylabel('mismatches per encoded tokens')
	plt.show()
