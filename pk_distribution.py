from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import serialization


def get_bits(data: bytes, length: int) -> [int]:
	bits = [0] * (8 * length)
	for i, byte in enumerate(data):
		for j in range(8):
			bits[i*8+j] = (byte >> j) & 1
	return bits

def add_all(stat: [int], data: [int]) -> [int]:
	assert len(data) == len(stat)
	for i, d in enumerate(data):
		stat[i] += d
	return stat

stat = [0] * (8 * 32)

for i in range(100000000):
	pk_data = X25519PrivateKey.generate().public_key().public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
	bits = get_bits(pk_data, 32)
	stat = add_all(stat, bits)
	print(i)
print(stat)
