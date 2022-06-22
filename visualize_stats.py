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
	#mismatches = list(map(lambda x: x['mismatches'], data))
	#print(mismatches)
	import matplotlib.pyplot as plt
	plt.plot(mismatch_stat, label = "mismatch rate")
	plt.ylabel('mismatches per encoded tokens')
	plt.show()
