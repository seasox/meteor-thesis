from coder import MeteorCoder
chosen_context = "Despite a long history of research and wide-spread applications to censorship resistant systems, practical steganographic systems capable of embedding messages into realistic communication distributions, like text, do not exist.\n\n"

message_text = "sample text"

# Use gpt2-medium for 345M param model
# Use gpt2-large for 774M param model
def get_model(seed=1234, model_name='gpt2', device='cuda'):
    import numpy as np
    import torch
    from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    enc = GPT2Tokenizer.from_pretrained(model_name)
    enc.unk_token = None
    enc.bos_token = None
    enc.eos_token = None
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    # model.double()  # want to avoid using this

    def decode(self, token_ids, **kwargs):
        filtered_tokens = self.convert_ids_to_tokens(token_ids)
        text = self.convert_tokens_to_string(filtered_tokens)
        return text
    GPT2Tokenizer.decode = decode

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, 0)
    GPT2Tokenizer._convert_token_to_id = _convert_token_to_id

    return enc, model


model_name = 'gpt2-medium'
device = 'cpu'

enc, model = get_model(model_name=model_name, device=device)
    
    
coder = MeteorCoder(enc, model, device)

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import serialization

# generate a private key pair
private_key = X25519PrivateKey.generate()
public_key = private_key.public_key()
public_key_data = public_key.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)

# encode public key with zero key
pk_key = b'\x00'*64
pk_nonce = b'\xfa'*64
encoded_pk = coder.encode_binary(public_key_data, chosen_context, pk_key, pk_nonce)

# send encoded_pk to bob

# Bob: generate key and send via stego channel
bobs_pk_data = X25519PrivateKey.generate().public_key().public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
bobs_encoded_pk = coder.encode_binary(bobs_pk_data, chosen_context, pk_key, pk_nonce)
# Alice receives Bob's PK
bobs_pk = X25519PublicKey.from_public_bytes(coder.decode_binary(bobs_encoded_pk[0], chosen_context, pk_key, pk_nonce))


# derive key
shared_key = private_key.exchange(bobs_pk)
derived_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b'handshake data').derive(shared_key)

nonce = b'\x01'*64

x = coder.encode_message(message_text, chosen_context, derived_key, nonce)
y = coder.decode_message(x[0], chosen_context, derived_key, nonce)
