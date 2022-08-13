import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F

# set seed used in PRG
seed = 42
np.random.seed(seed)

# initialize tokenizer and model for GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

context_tokens = tokenizer.encode("I'm a language model")
context = torch.tensor([tokenizer.encoder['<|endoftext|>']] + context_tokens, dtype=torch.long)
prev = context
output = context
past = None
max_len = 43
temperature = 0.7
for _ in range(max_len):
	# get model result for next token
	result = model(prev.unsqueeze(0), past_key_values=past)
	logits = result.logits
	past = result.past_key_values
	# preprocessing logits: sort, convert to double, scale by temperature
	logits, indices = logits[0, -1, :].sort(descending=True)
	logits = logits.double()
	logits_temp = logits / temperature
	# apply the softmax function to convert logits to probability distribution
	probs_temp = F.softmax(logits_temp, dim=0)
	# accumulate probabilites
	cum_probs = probs_temp.cumsum(0)
	# sample from probabilites
	sample = np.random.random_sample()
	selection = (cum_probs > sample).nonzero()[0].item()

	# append selected token to output
	prev = indices[selection].view(1)
	output = torch.cat((output, prev))
	# stop if endoftext was sampled
	if prev == tokenizer.encoder['<|endoftext|>']:
		break

print(tokenizer.decode(output, skip_special_tokens=True))
