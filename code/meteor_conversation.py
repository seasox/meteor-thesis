import logging
import os
import sys

from util import get_model
from coder import MeteorCoder


def main():
    model_name = 'gpt2-medium'
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
    history = [
        "Hi! What's your favorite movie?",  # Bob
        "Hey there :) my favorite movie is Harry Potter. I like the second part the most, but the fourth is also good. And yours?",
        "Hmm, I like Lord Of The Rings. I've also watched and read Harry Potter, but the story didn't quite work for me. Not that much of a wizard fan",
        "Oh you should give it a try again! HP is not THAT much about wizardry actually. It's a movie about friendship.",
        "Oh is it? I thought they have speaking snakes and witches and goblins and thelike",
        "Yes, it's about the wizarding world, but the things they do are actually about companionship and love",
        "How that?",
        "Well, just take Harry for example: he's an orphan who was raised without love, but when he came to Hogwarts, he was loved everywhere.",
    ]
    message_text = "Hi! Did anyone follow you last night? Are we still up for tommorow? It was 12 am at the market, right?"
    key = os.urandom(64)
    nonce = os.urandom(64)
    reconst = ''
    while True:
        if message_text == '':
            break
        bob_says = input('Please enter your message: ')
        print("Bob: " + bob_says)
        history += [bob_says]
        stegotext, tokens, stats, message_text = coder.encode_conversation(message_text, history, key, nonce)
        print('Alice: ' + stegotext)
        recovered, tokens = coder.decode_conversation(stegotext, history, key, nonce)
        reconst += recovered
        history += [stegotext]
    print("*"*16 + " RECONST " + "*"*16)
    print(reconst)
    print("*" * 16 + " CONVERSATION " + "*" * 16)
    print(coder.conversation_context_from_history(history))


if __name__ == '__main__':
    main()
