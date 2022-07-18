import base64
import time

from bitarray import bitarray
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from coder import MeteorCoder, encode_context
from util import get_model

if __name__ == '__main__':
    # chosen_context_str = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"
    chosen_context_str = "The following text is a conversation between two friends named Sebastian and Anna. They have been engaged for two years and are planning their vacation together, which they plan to spend in Moscow next month:\n\n"
    model_name = 'gpt2-medium'
    device = 'cpu'

    enc, model = get_model(model_name=model_name, device=device)

    chosen_context = encode_context(chosen_context_str, enc)

    coder = MeteorCoder(enc, model, device)

    # generate a DSA key pair

    alice_sign = Ed25519PrivateKey.generate()
    alice_verify = alice_sign.public_key()

    # generate a DH private key pair
    alice_sk = X25519PrivateKey.generate()
    alice_pk = alice_sk.public_key()
    alice_pk_data = alice_pk.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)

    alice_signature: bytes = alice_sign.sign(alice_pk_data)  # 64 bytes

    assert len(alice_pk_data) == 32
    assert len(alice_signature) == 64

    alice_verify.verify(alice_signature, alice_pk_data)

    print('Alice\'s public key (base64):')
    print(base64.b64encode(alice_pk_data))

    print('Encoding Alice\'s public key to send...')
    # encode public key with zero key

    # pk_encryption = PRGEncryption(DRBG(pk_key, sample_seed_prefix + pk_nonce))
    public_key_data_ba = bitarray()
    public_key_data_ba.frombytes(alice_pk_data + alice_signature)

    # encoded_pk = coder.encode_binary(public_key_data_ba.tolist(), chosen_context, pk_encryption)

    # send encoded_pk to bob
    # print('Send Alice\'s PK to Bob (TODO)')

    print('Encoding Bob\'s public key to send...')
    # Bob: generate key and send via stego channel
    bob_sign = Ed25519PrivateKey.generate()
    bob_verify = bob_sign.public_key()
    bob_sk = X25519PrivateKey.generate()
    bob_pk = bob_sk.public_key()
    bob_pk_data = bob_pk.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    bob_signature = bob_sign.sign(bob_pk_data)

    assert len(bob_pk_data) == 32
    assert len(bob_signature) == 64
    bob_verify.verify(bob_signature, bob_pk_data)

    bobs_pk_data_ba = bitarray()
    bobs_pk_data_ba.frombytes(bob_pk_data + bob_signature)

    sample_seed_prefix = b'sample'
    pk_key = b'\x00' * 64
    pk_nonce = b'\xfa' * 64
    bob_encoded_pk = coder.encode_binary(bobs_pk_data_ba.tolist(), chosen_context, pk_key, pk_nonce)
    print('Send Bob\'s PK to Alice')
    # Alice receives Bob's PK
    bob_msg_recv = bitarray(coder.decode_binary(bob_encoded_pk[0], chosen_context, pk_key, pk_nonce)[:96 * 8]).tobytes()
    bob_pk_data_recv = bob_msg_recv[:32]
    bob_signature_recv = bob_msg_recv[32:96]

    bob_verify.verify(bob_signature_recv, bob_pk_data_recv)  # this throws InvalidSignature on failure

    bob_pk_recv = X25519PublicKey.from_public_bytes(bob_pk_data_recv)
    print('Received Bob\'s PK')

    assert bob_pk_recv.public_bytes(encoding=serialization.Encoding.Raw,
                                    format=serialization.PublicFormat.Raw) == bob_pk.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)

    print('Derive key')
    # derive key
    shared_key = alice_sk.exchange(bob_pk_recv)

    assert shared_key == alice_sk.exchange(bob_pk)

    derived_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'handshake data').derive(shared_key)

    # Constants for HMAC-DRBG -- MUST CHANGE FOR SECURE IMPLEMENTATION
    sample_nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

    print('Encode message using derived key')
    message_text = "Hi! Did anyone follow you last night?"
    start = time.time()
    x = coder.encode_message(message_text, chosen_context_str, derived_key, sample_nonce)
    y = coder.decode_message(x[0], chosen_context_str, derived_key, sample_nonce)
    end = time.time()
    print(end - start)
    assert y == message_text
    chosen_context_str += x[0]
    chosen_context_str += '\n\n'
    message_text = "No, I\'m fine"
    x = coder.encode_message(message_text, chosen_context_str, derived_key, sample_nonce)
    y = coder.decode_message(x[0], chosen_context_str, derived_key, sample_nonce)
    assert y == message_text
