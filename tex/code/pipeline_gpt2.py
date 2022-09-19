generator = pipeline("text-generation", model='gpt2-large')

print(generator("What is steganography in computer science?", max_length=128, do_sample=True)[0]['generated_text'])
