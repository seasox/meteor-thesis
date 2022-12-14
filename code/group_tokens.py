from util import get_model


def main():
	enc, model = get_model(device='mps')
	print(f'Tokenizer has {len(enc.encoder)} tokens')
	grouped_keys = []
	keys = enc.encoder.keys()
	stats = {}
	# keys = [ 'hello', 'hel', 'lo', 'h' ]
	for k1 in keys:
		did_add = False
		for k2 in grouped_keys:
			if k1 != k2 and k2.startswith(k1):
				grouped_keys.append(k1)
				grouped_keys.remove(k2)
				del stats[k2]
				stats[k1] = 1
				print('ye')
				did_add = True
				break
			if k1 != k2 and k1.startswith(k2):
				did_add = True
				stats[k2] += 1
				break
		if not did_add:
			grouped_keys.append(k1)
			stats[k1] = 1
	print(grouped_keys)
	print(f'stats: {sorted(stats.items(), key=lambda i: i[1])}')


if __name__ == '__main__':
	main()
