from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

from coder import MeteorCoder
import cryptography.hazmat.primitives.asymmetric.rsa as rsa

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

enc_text = encrypted_message.decode("ASCII", 'ignore')

# encrypted_message probably is not uniformly distributed and might leak information to Warden

# send encrypted_message to Bob

key = b'\00'*64
nonce = b'\x01'*64

x = coder.encode_message(enc_text, chosen_context, key, nonce)
recovered_encrypted_message = coder.decode_message(x[0], chosen_context, key, nonce)

decrypted_message = bob_private_key.decrypt(recovered_encrypted_message, pad)

assert decrypted_message == message_text
