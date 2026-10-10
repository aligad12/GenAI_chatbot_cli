import argparse
from langchain_core.messages import HumanMessage
from llm import get_llm


def main():
    parser = argparse.ArgumentParser(description="Simple Groq chat CLI")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-stream", action="store_true", help="disable streaming")
    args = parser.parse_args()

    streaming = not args.no_stream
    llm = get_llm(temperature=args.temperature, streaming=streaming)

    print(f"Chat started (temperature={args.temperature}, streaming={streaming}). Type 'exit' to quit.")
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        print("AI: ", end="", flush=True)
        if streaming:
            for chunk in llm.stream([HumanMessage(content=user)]):
                print(chunk.content, end="", flush=True)
            print()
        else:
            print(llm.invoke([HumanMessage(content=user)]).content)


if __name__ == "__main__":
    main()