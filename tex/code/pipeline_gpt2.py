generator = pipeline("text-generation", model='gpt2')

print("Example of GPT-2 model output")
print(generator("Hello, I'm a language model,", max_length=43, do_sample=True)[0]['generated_text'])
