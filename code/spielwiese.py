import os

from coder import encrypt_ctr_aes, decrypt_ctr_aes

if __name__ == '__main__':
    while True:
        key = os.urandom(32)
        cipher = encrypt_ctr_aes(key, 'hello'.encode())
        print(cipher)
        message = decrypt_ctr_aes(key, cipher)
        print(message.decode())
