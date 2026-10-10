import math
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from llm import get_llm


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression such as 'sqrt(144)' or '2**10 + 5'. Supports math functions like sqrt, sin, cos, log and pi."""
    allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: {e}"


@tool
def wikipedia(query: str) -> str:
    """Look up a topic on Wikipedia and return a short summary."""
    return f"[Fixed output] Wikipedia summary for '{query}': This is a placeholder article text."


TOOLS = {"calculator": calculator, "wikipedia": wikipedia}


def manual_calls():
    print("=== 1) Manual tool calls ===")
    print("calculator('sqrt(144)') ->", calculator.invoke({"expression": "sqrt(144)"}))
    print("calculator('2**10 + 5') ->", calculator.invoke({"expression": "2**10 + 5"}))
    print("wikipedia('Ada Lovelace') ->", wikipedia.invoke({"query": "Ada Lovelace"}))


def model_with_tools():
    print("\n=== 2) Tools bound to the model ===")
    llm = get_llm(temperature=0).bind_tools(list(TOOLS.values()))

    questions = [
        "What is the square root of 1764?",
        "Tell me about Ada Lovelace.",
        "Say hello in one short sentence.",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        messages = [HumanMessage(content=q)]
        ai_msg = llm.invoke(messages)
        print("Tool calls requested:", ai_msg.tool_calls)

        if not ai_msg.tool_calls:
            print("A:", ai_msg.content)
            continue

        messages.append(ai_msg)
        for call in ai_msg.tool_calls:
            tool_message = TOOLS[call["name"]].invoke(call)  # returns a ToolMessage
            print(f"Tool result ({call['name']}):", tool_message.content)
            messages.append(tool_message)

        final = llm.invoke(messages)
        print("A:", final.content)


if __name__ == "__main__":
    manual_calls()
    model_with_tools()