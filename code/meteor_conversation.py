import logging
import os

from coder import MeteorCoder
from util import get_model


def main():
    model_name = 'gpt2-medium'
    #model_name = 'microsoft/DialoGPT-large'
    device = 'cpu'

    print('get model')
    logging.basicConfig(level=logging.DEBUG)
    enc, model = get_model(model_name=model_name, device=device)
    print('done')

    coder = MeteorCoder(enc, model, device)

    # Constants for HMAC-DRBG -- MUST CHANGE FOR SECURE IMPLEMENTATION
    key = b'0x01' * 64
    nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    """history = [
        "Hi! What's your name?",  # Bob
        "My name is <sp2>. And yours?",  # Alice
        "I'm <sp1>. How are you today?",  # Bob
        "I'm doing well, thanks.",  # Alice
        "What are your hobbies?",  # Bob
        "I like dancing and swimming. But I work a lot, so I don't get to do these a lot. What's your hobby?",  # Alice
        "I don't have any to be honest. I'm very busy with the house. Sometimes, I just sit in the garden and look at the sunset.",  # Bob
        "That's also a kind of hobby, isn't it? I think we have a lot in common! Maybe we should go for a swim some time.",  # Alice
        "That would be great!",  # Bob
        "Hey! What's your plan for today?",  # Alice
    ]"""
    history = [ "Hi! How are you?",
                "Fine, you?",
                "I'm alright. It's pretty hot outside today haha",
                "Yeah, here too"]
    """history = [
        "Hi! What's your favorite movie?",  # Bob
        "Hey there :) my favorite movie is Harry Potter. I like the second part the most, but the fourth is also good. And yours?",
        "Hmm, I like Lord Of The Rings. I've also watched and read Harry Potter, but the story didn't quite work for me. Not that much of a wizard fan",
        "Oh you should give it a try again! HP is not THAT much about wizardry actually. It's a movie about friendship.",
        "Oh is it? I thought they have speaking snakes and witches and goblins and thelike",
        "Yes, it's about the wizarding world, but the things they do are actually about companionship and love",
        "How that?",
        "Well, just take Harry for example: he's an orphan who was raised without love, but when he came to Hogwarts, he was loved everywhere.",
    ]"""
    #history = []
    #message_text = "Hi! Did anyone follow you last night? Are we still up for tommorow? It was 12 am at the market, right?"
    message_text = "Hi there!"
    # message_ctx = [enc.encoder['<|endoftext|>']]
    # message = codr.decode_arithmetic(
    #    model, enc, message_text, message_ctx, precision=40, topk=60000, device=device)
    import bitarray
    message = bitarray.bitarray()
    message.fromstring(message_text)
    message = message.tolist()
    key = os.urandom(64)
    nonce = os.urandom(64)
    print("session key: ", key)
    print("session nonce: ", nonce)
    reconst = []
    remainder = message
    while True:
        if not remainder:
            break
        chunk_length = 32
        stegotext, remainder = coder.encode_conversation(remainder, history, key, nonce, max_length=chunk_length)
        print('Alice: ' + stegotext)
        recovered, tokens = coder.decode_conversation(stegotext, history, key, nonce)
        reconst += recovered[:chunk_length]
        history += [stegotext]
        bob_says = input('Please enter your message: ')
        # bob_says = "But who is your favorite character?"
        print("Bob: " + bob_says)
        history += [bob_says]
        print(f'{len(remainder)} bits left')
        print("history: %s" % ' -- '.join(history))
    reconst = bitarray.bitarray(reconst)
    reconst = reconst.tobytes().decode('utf-8', 'replace')
    # reconst = codr.encode_arithmetic(model, enc, reconst, message_ctx, precision=40, topk=60000, device=device)
    print("*" * 16 + " RECONST " + "*" * 16)
    print(reconst)
    print("*" * 16 + " CONVERSATION " + "*" * 16)
    print('\n'.join(history))


if __name__ == '__main__':
    main()
