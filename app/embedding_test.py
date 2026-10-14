import ollama

text = "VNet peering allows two Azure virtual networks to communicate."

response = ollama.embed(
    model="nomic-embed-text",
    input=text
)

embedding = response["embeddings"][0]

print("Text:")
print(text)

print("\nNumber of dimensions:")
print(len(embedding))

print("\nFirst 10 numbers:")
print(embedding[:10])