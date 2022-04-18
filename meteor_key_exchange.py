from coder import MeteorCoder, DRBG
from meteor_symmetric import PRGEncryption
from util import get_model
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import serialization
import base64

chosen_context = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"
message_text = "sample text"
model_name = 'gpt2-medium'
device = 'cpu'

enc, model = get_model(model_name=model_name, device=device)

coder = MeteorCoder(enc, model, device)

# generate a private key pair
private_key = X25519PrivateKey.generate()
public_key = private_key.public_key()
public_key_data = public_key.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)


print('Alice\'s public key (base64):')
print(base64.b64encode(public_key_data))

print('Encoding Alice\'s public key to send...')
# encode public key with zero key
sample_seed_prefix = b'sample'
pk_key = b'\x00' * 64
pk_nonce = b'\xfa' * 64
pk_encryption = PRGEncryption(DRBG(pk_key, sample_seed_prefix + pk_nonce))
pk_decryption = PRGEncryption(DRBG(pk_key, sample_seed_prefix + pk_nonce))
encoded_pk = coder.encode_binary(public_key_data, chosen_context, pk_encryption)

# send encoded_pk to bob
print('Send Alice\'s PK to Bob')

print('Encoding Bob\'s public key to send...')
# Bob: generate key and send via stego channel
sample_seed_prefix = b'sample'
sample_key = b'0x00' * 64
sample_nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
pk_encryption = PRGEncryption(DRBG(sample_key, sample_seed_prefix + sample_nonce))
pk_decryption = PRGEncryption(DRBG(sample_key, sample_seed_prefix + sample_nonce))
bobs_pk_data = X25519PrivateKey.generate().public_key().public_bytes(encoding=serialization.Encoding.Raw,
                                                                     format=serialization.PublicFormat.Raw)
bobs_encoded_pk = coder.encode_binary(bobs_pk_data, chosen_context, pk_encryption)
print('Send Bob\'s PK to Alice')
# Alice receives Bob's PK
bobs_pk = X25519PublicKey.from_public_bytes(coder.decode_binary(bobs_encoded_pk[0], chosen_context, pk_decryption))
print('Received Bob\'s PK')

print('Derive key')
# derive key
shared_key = private_key.exchange(bobs_pk)
derived_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'handshake data').derive(shared_key)

# Constants for HMAC-DRBG -- MUST CHANGE FOR SECURE IMPLEMENTATION
sample_seed_prefix = b'sample'
sample_key = b'0x01' * 64
sample_nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

print('Encode message using derived key')
encryption = PRGEncryption(DRBG(sample_key, sample_seed_prefix + sample_nonce))
decryption = PRGEncryption(DRBG(sample_key, sample_seed_prefix + sample_nonce))
x = coder.encode_message(message_text, chosen_context, encryption)
y = coder.decode_message(x[0], chosen_context, encryption)
