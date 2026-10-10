from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from llm import get_llm

def main():
    llm = get_llm(temperature=0)
    memory = ConversationBufferMemory(return_messages=True, memory_key="history")
    system = SystemMessage(content="you are a helpful assistant. Keep answers short.")


    turns = [
         "Hi, my name is Alex and I'm building a LangChain project.",
        "What is my name?",
        "What am I building?",
    ]

    for user_input in turns:
        history = memory.load_memory_variables({})["history"]
        messages = [system, *history, HumanMessage(content=user_input)]

        response = llm.invoke(messages)

        memory.save_context({"input": user_input}, {"output": response.content})
        print(f"You: {user_input}\nAI:  {response.content}\n")

    print("--- Final buffer contents ---")
    for m in memory.load_memory_variables({})["history"]:
        print(f"[{m.type}] {m.content}")


if __name__ == "__main__":
    main()