from transformers import pipeline, set_seed

generator = pipeline("text-generation", model='gpt2')
set_seed(42)

print("Example of GPT-2 model output without sampling")
print(generator("Hello, I'm a language model,", max_length=43, do_sample=False)[0]['generated_text'])

print("Example of GPT-2 model output with sampling")
print(generator("Hello, I'm a language model,", max_length=43, do_sample=True)[0]['generated_text'])
