from tokenizer import SimpleTokenizer


training_text = """
the cat sat on the mat
the cat is happy
the dog sat on the mat
"""


tokenizer = SimpleTokenizer(training_text)


print("Vocabulary:")
print(tokenizer.token_to_id)


sentence = "the cat sat"

encoded = tokenizer.encode(sentence)

print("\nOriginal sentence:")
print(sentence)

print("\nEncoded:")
print(encoded)

decoded = tokenizer.decode(encoded)

print("\nDecoded:")
print(decoded)