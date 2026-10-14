import ollama

messages = []

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": question
    })

    response = ollama.chat(
        model="llama3.2:3b",
        messages=messages
    )

    answer = response["message"]["content"]

    messages.append({
        "role": "assistant",
        "content": answer
    })

    print("\nAI:", answer)