import bitarray
import cryptography.hazmat.primitives.asymmetric.rsa as rsa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

import meteor_symmetric
from coder import MeteorCoder, DRBG
from util import get_model

"""
class RsaStegoCrypto:
    def __init__(self):
        pass

    def arr_xor(self, a, b):
        out = a.copy()
        for i in range(len(out)):
            out = out[i] ^ b[i]
        return out

    def encrypt(self, m: bytes, N: long, e: long):
        k = N.bit_length()
        l = len(m)
        x = [0] * (l+1)
        b = [0] * (l+1)
        while True:
            x[0] = random.randint(1, N)  # top exclusive
            for i in range(1, l+1):
                b[i] = x[i-1] % 2
                x[i] = pow(x[i-1], e, N)
            c = numpy.random.randint(0, 2)
            if x[l] <= (2**k-N) or c == 1:
                break
        xd = None
        if (x[1] <= 2 ** k - N) and c == 0:
            xd = x
        elif (x[1] <= 2 ** k - N) and c == 1:
            xd = 2 ** k - x
        assert xd is not None

        return xd, self.arr_xor(m, b)


    def decrypt(self, xd, c, N, d):
        l = len(c)
        k = N.bit_length()
        x = [0] * (l+1)
        if xd > N:
            x[l] = xd
        else:
            x[l] = 2 ** k - xd

        b = [None] * l
        for i in range(l, 0, -1): #  bottom exclusive
            x[i-1] = pow(x[i], d, N)
            b[i] = x[i-1] % 2

        return self.arr_xor(c, b)
"""

if __name__ == '__main__':
    chosen_context_str = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"

    message_text = "sample text"

    model_name = 'gpt2-medium'
    device = 'cpu'

    enc, model = get_model(model_name=model_name, device=device)

    chosen_context = enc.encode(chosen_context_str)

    coder = MeteorCoder(enc, model, device)

    # generate a private key pair
    bob_private_key = rsa.generate_private_key(65537, 2048)

    # we expect this public key to be available to Alice
    bob_public_key = bob_private_key.public_key()

    # apply OAEP padding
    pad = padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None)

    # encrypt message using Bob's public key
    encrypted_message: bytes = bob_public_key.encrypt(message_text.encode('UTF-8'), pad)
    encrypted_message_ba = bitarray.bitarray()
    encrypted_message_ba.frombytes(encrypted_message)

    # encrypted_message probably is not uniformly distributed and might leak information to Warden

    # send encrypted_message to Bob

    sample_seed_prefix = b'sample'
    sample_key = b'0x01' * 64
    sample_nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    encryption = meteor_symmetric.PRGEncryption(DRBG(sample_key, sample_seed_prefix + sample_nonce))
    decryption = meteor_symmetric.PRGEncryption(DRBG(sample_key, sample_seed_prefix + sample_nonce))
    x = coder.encode_binary(encrypted_message_ba.tolist(), chosen_context, encryption)
    recovered_encrypted_message = coder.decode_binary(x[0], chosen_context, decryption)

    decrypted_message = bob_private_key.decrypt(bitarray.bitarray(recovered_encrypted_message).tobytes()[:256],
                                                pad).decode('UTF-8')

    assert decrypted_message == message_text
