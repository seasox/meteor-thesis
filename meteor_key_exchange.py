from bitarray import bitarray

from coder import MeteorCoder, DRBG, encode_context
from meteor_symmetric import PRGEncryption
from util import get_model
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import serialization
import base64

chosen_context_str = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"
message_text = "sample text"
model_name = 'gpt2-medium'
device = 'cpu'

enc, model = get_model(model_name=model_name, device=device)

chosen_context = encode_context(chosen_context_str, enc)

coder = MeteorCoder(enc, model, device)

# generate a private key pair
alice_sk = X25519PrivateKey.generate()
alice_pk = alice_sk.public_key()
alice_pk_data = alice_pk.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)


print('Alice\'s public key (base64):')
print(base64.b64encode(alice_pk_data))

print('Encoding Alice\'s public key to send...')
# encode public key with zero key

# pk_encryption = PRGEncryption(DRBG(pk_key, sample_seed_prefix + pk_nonce))
public_key_data_ba = bitarray()
public_key_data_ba.frombytes(alice_pk_data)

# encoded_pk = coder.encode_binary(public_key_data_ba.tolist(), chosen_context, pk_encryption)

# send encoded_pk to bob
# print('Send Alice\'s PK to Bob (TODO)')

print('Encoding Bob\'s public key to send...')
# Bob: generate key and send via stego channel
bob_sk = X25519PrivateKey.generate()
bob_pk = bob_sk.public_key()
bobs_pk_data_ba = bitarray()
bobs_pk_data_ba.frombytes(bob_pk.public_bytes(encoding=serialization.Encoding.Raw,
                                              format=serialization.PublicFormat.Raw))
sample_seed_prefix = b'sample'
pk_key = b'\x00' * 64
pk_nonce = b'\xfa' * 64
pk_encryption = PRGEncryption(DRBG(pk_key, sample_seed_prefix + pk_nonce))
pk_decryption = PRGEncryption(DRBG(pk_key, sample_seed_prefix + pk_nonce))
bob_encoded_pk = coder.encode_binary(bobs_pk_data_ba.tolist(), chosen_context, pk_encryption)
print('Send Bob\'s PK to Alice')
# Alice receives Bob's PK
bob_pk_data_recv = bitarray(coder.decode_binary(bob_encoded_pk[0], chosen_context, pk_decryption)[:32*8]).tobytes()
bob_pk_recv = X25519PublicKey.from_public_bytes(bob_pk_data_recv)
print('Received Bob\'s PK')

assert bob_pk_recv.public_bytes(encoding=serialization.Encoding.Raw,
                                              format=serialization.PublicFormat.Raw) == bob_pk.public_bytes(encoding=serialization.Encoding.Raw,
                                              format=serialization.PublicFormat.Raw)

print('Derive key')
# derive key
shared_key = alice_sk.exchange(bob_pk_recv)

assert shared_key == alice_sk.exchange(bob_pk)

derived_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'handshake data').derive(shared_key)

# Constants for HMAC-DRBG -- MUST CHANGE FOR SECURE IMPLEMENTATION
sample_seed_prefix = b'sample'
sample_key = b'0x01' * 64
sample_nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

print('Encode message using derived key')
encryption = PRGEncryption(DRBG(derived_key, sample_seed_prefix + sample_nonce))
decryption = PRGEncryption(DRBG(derived_key, sample_seed_prefix + sample_nonce))
x = coder.encode_message(message_text, chosen_context_str, encryption)
y = coder.decode_message(x[0], chosen_context_str, decryption)
