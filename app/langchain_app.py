from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="llama3.2:3b"
)

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} to me like I'm a complete beginner. "
    "Use a simple real-world analogy and an Azure example."
)

chain = prompt | llm

response = chain.invoke({
    "topic": "Azure Virtual Network"
})

print(response.content)