def get_model(seed=1234, model_name='gpt2', device='cuda'):
	import numpy as np
	import torch
	from typing import Tuple, List
	from transformers import GPT2LMHeadModel, GPT2Tokenizer
	np.random.seed(seed)
	torch.random.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	enc = GPT2Tokenizer.from_pretrained(model_name)
	# enc.unk_token = None
	# enc.bos_token = None
	# enc.eos_token = None

	# check no token has inner space (which holds for subword tokenizers)
	assert len(list(filter(lambda x: x.find('Ġ') > 0, enc.encoder.keys()))) == 0

	model = GPT2LMHeadModel.from_pretrained(model_name)
	# model.resize_token_embeddings(len(enc))
	model.to(device)
	model.eval()

	# model.double()  # want to avoid using this

	def decode(self, token_ids, **kwargs) -> Tuple[str, List[str]]:
		filtered_tokens = self.convert_ids_to_tokens(token_ids, kwargs)
		text = self.convert_tokens_to_string(filtered_tokens)
		return text, filtered_tokens

	GPT2Tokenizer.decode = decode

	def _convert_token_to_id(self, token):
		return self.encoder.get(token, 0)

	GPT2Tokenizer._convert_token_to_id = _convert_token_to_id

	def prepare_model_inputs(self: GPT2LMHeadModel, inputs: torch.Tensor, num_return_sequences=None, bos_token_id=None,
							 output_attentions=None, output_hidden_states=None, use_cache=None, **model_kwargs):
		bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
		output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
		output_hidden_states = (
			output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
		)
		num_return_sequences = (
			num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
		)
		# 2. Define model inputs
		# inputs_tensor has to be defined
		# model_input_name is defined if model-specific keyword input is passed
		# otherwise model_input_name is None
		# all model-specific keyword inputs are removed from `model_kwargs`
		inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
		batch_size = inputs_tensor.shape[0]

		# expect decoder-only
		input_ids = inputs_tensor

		# 3. Define other model kwargs
		model_kwargs["output_attentions"] = output_attentions
		model_kwargs["output_hidden_states"] = output_hidden_states
		model_kwargs["use_cache"] = use_cache
		# 11. expand input_ids with `num_return_sequences` additional sequences per batch
		input_ids, model_kwargs = self._expand_inputs_for_generation(
			input_ids,
			expand_size=num_return_sequences,
			is_encoder_decoder=self.config.is_encoder_decoder,
			**model_kwargs,
		)
		return input_ids, model_kwargs

	GPT2LMHeadModel.prepare_model_inputs = prepare_model_inputs

	return enc, model


def main():
	enc, model = get_model(device='mps')
	print(f'Tokenizer has {len(enc.encoder)} tokens')
	grouped_keys = []
	keys = enc.encoder.keys()
	stats = {}
	# keys = [ 'hello', 'hel', 'lo', 'h' ]
	for k1 in keys:
		did_add = False
		for k2 in grouped_keys:
			if k1 != k2 and k2.startswith(k1):
				grouped_keys.append(k1)
				grouped_keys.remove(k2)
				del stats[k2]
				stats[k1] = 1
				print('ye')
				did_add = True
				break
			if k1 != k2 and k1.startswith(k2):
				did_add = True
				stats[k2] += 1
				break
		if not did_add:
			grouped_keys.append(k1)
			stats[k1] = 1
	print(grouped_keys)
	print(f'stats: {sorted(stats.items(), key=lambda i: i[1])}')


if __name__ == '__main__':
	main()
