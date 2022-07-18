from transformers import AutoModelForCausalLM, AutoTokenizer

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


def conversation_context_from_history(eos_token, history: [str]):
    history = history + ['']  # add empty message to end of history (this is the start of the answer
    return eos_token.join([msg for msg in history])


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

history = conversation_context_from_history(tokenizer.eos_token, history)

chat_history_ids = tokenizer.encode(history, return_tensors='pt')

# Let's chat for 5 lines
# for step in range(5):
while True:
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    # inp = input(">> User:")
    inp = "Hey! How are you?"
    new_user_input_ids = tokenizer.encode(inp + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    # bot_input_ids = torch.cat((chat_history_ids, new_user_input_ids), dim=-1)
    # bot_input_ids = torch.cat((chat_history_ids, new_user_input_ids), dim=-1) if step > 0 else new_user_input_ids
    # bot_input_ids = torch.cat((chat_history_ids, new_user_input_ids), dim=-1) if chat_history_ids is not None else new_user_input_ids
    bot_input_ids = new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id,
                                      do_sample=True)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(
        tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
