from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


def get_bits(data: bytes) -> [int]:
	return [int(i) for i in ''.join(['{:08b}'.format(b) for b in data])]

def add_all(stat: [int], data: [int]) -> [int]:
	assert len(data) == len(stat)
	for i, d in enumerate(data):
		stat[i] += d
	return stat

stat = [0] * (8 * 32)

derived_key_stat = [0] * (8 * 32)

for i in range(10**6):
	sk = X25519PrivateKey.generate()
	pk = X25519PrivateKey.generate().public_key()
	pk_data = pk.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
	shared_key = sk.exchange(pk)
	derived_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'handshake data').derive(shared_key)
	pk_bits = get_bits(pk_data)
	derived_key_bits = get_bits(derived_key)
	stat = add_all(stat, pk_bits)
	derived_key_stat = add_all(derived_key_stat, derived_key_bits)
	print(i)

from scipy.stats import chisquare

# remove MSB from stat
assert stat.pop(len(stat)-8) == 0
print(stat)
print(chisquare(stat))
print(chisquare(derived_key_stat))
