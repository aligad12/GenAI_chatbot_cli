# LangChain Assignment (Groq)

Hands-on LangChain assignment covering chat models, message schemas, conversation memory, few-shot prompt templates, and tool calling.

> **Note:** The assignment references OpenAI's `ChatOpenAI`. This project uses **Groq** (`ChatGroq` from `langchain-groq`) instead, since Groq offers a free API tier. The LangChain concepts are identical; only the model class changes.

## Project Structure

```
langchain-assignment/
├── .env                      # your API key (not committed)
├── .env.example              # template for .env
├── .gitignore
├── README.md
├── requirements.txt
├── llm.py                    # shared Groq model helper
├── list_models.py            # prints the Groq models your key can use
├── 1_chat_cli.py             # 1.1 ChatModels: CLI chat
├── 2_1_messages.py           # 2.1 Messages: three-turn template + SystemMessage
├── 2_2_memory.py             # 2.2 Messages: ConversationBufferMemory
├── 3_few_shot_sentiment.py   # 3.1 PromptTemplate: few-shot sentiment classifier
└── 4_tool_calling.py         # 4.1 ToolCalling: calculator + Wikipedia tools
```

## Setup

1. **Clone the repo and enter it**
   ```bash
   git clone https://github.com/<your-username>/langchain-assignment.git
   cd langchain-assignment
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get a free Groq API key** at <https://console.groq.com/keys>.

5. **Create a `.env` file** (copy `.env.example`) and fill it in:
   ```
   GROQ_API_KEY=your_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   ```
   Groq's model lineup changes over time. Run `python list_models.py` to see the model IDs currently available to your key, and set `GROQ_MODEL` to any chat model from that list.

## How to Run

| Task | Command |
|------|---------|
| 1.1 Chat CLI | `python 1_chat_cli.py` (options: `--temperature 0.2`, `--no-stream`) |
| 2.1 Messages | `python 2_1_messages.py` |
| 2.2 Memory | `python 2_2_memory.py` |
| 3.1 Few-shot sentiment | `python 3_few_shot_sentiment.py` |
| 4.1 Tool calling | `python 4_tool_calling.py` |

---

## 1. ChatModels: CLI Chat (`1_chat_cli.py`)

A command-line chat built on `ChatGroq`. It exposes two model parameters:

- **`temperature`** controls randomness (0 is near-deterministic, higher values give more varied output). Set with `--temperature`.
- **`streaming`** prints tokens as they arrive using `llm.stream()`. Disable with `--no-stream`.

Each message is sent independently. Conversation memory is covered in section 2.2.

**Sample output:**
```
(.venv) PS C:\Users\Lenovo\OneDrive\Desktop\GenAI\langchain-assignment> python .\1_chat_cli.py
Chat started (temperature=0.7, streaming=True). Type 'exit' to quit.

You: tell me about F1 shortly
AI: **Formula 1 (F1)** – the pinnacle of single‑seat, open‑wheel racing

| What it is | Key facts |
|------------|-----------|
| **Sport** | International motor‑sport championship governed by the FIA. |
| **Cars** | 10‑seat, 1‑engine, 4‑wheel cars built to extremely tight technical regulations. |
| **Season** | 20‑22 Grands Prix (races) on circuits around the world, from street tracks to permanent raceways. |
| **Teams** | 10 factory teams (e.g., Mercedes‑Petronas, Red Bull Racing, Ferrari) and a few customer‑entry teams. |
| **Drivers** | 20 professional drivers (2 per team) compete for the World Drivers’ Championship. |
| **Points** | Awarded to the top 10 finishers (25‑1 points). The driver and constructor with the most points at the end of the season win the titles. |
| **Technology** | Cutting‑edge hybrid power‑units (V6 turbo‑charged + energy‑recovery system), aerodynamic research, and advanced materials. |
| **Audience** | Global viewership in the billions; a cultural phenomenon with a huge fan base, media coverage, and a huge economic impact on host cities. |

In short, Formula 1 is the fastest, most technologically advanced, and globally most popular form of motor racing, featuring elite drivers, high‑budget teams, and a championship that blends speed, strategy, and engineering innovation.

You: quit
```

## 2.1 Messages: Three-Turn Template (`2_1_messages.py`)

Uses LangChain's message schemas (`SystemMessage`, `HumanMessage`, `AIMessage`) to build a conversation of the form:

```
SystemMessage -> Human -> AI -> Human -> AI -> Human (model answers this one)
```

The same conversation is run twice with different system prompts (a concise engineer and a pirate) to show how the system message changes the output.

### How the SystemMessage influences the model

The `SystemMessage` is the first message in the conversation. It sets the model's role, tone, rules, and output format. The model conditions every later reply on it, even though the user never sees it. In this experiment the user questions were identical in both runs, but the responses differed in style and wording because the system prompt differed. In short, the `SystemMessage` steers *how* the model answers, while the `HumanMessage` carries *what* is being asked.

**Sample output:**
```
(.venv) PS C:\Users\Lenovo\OneDrive\Desktop\GenAI\langchain-assignment> python .\2_1_messages.py

=== SystemMessage: Concise engineer ===
[system] You are a senior automotive engineer. Answer in at most 2 sentences.
[human] What is a Formula Student car?
[ai] It's a small open-wheel race car designed and built by university students for competition.
[human] Why is the suspension important?
[ai] It keeps the tyres in contact with the road, which controls grip, handling, and driver comfort.
[human] How would a team tune it for a tight autocross track?

[ai response] For a tight autocross, the team would stiffen the springs and dampers, lower the ride height, and increase negative camber to maximize tyre contact during quick, short turns, while tuning the anti‑roll bars to balance front‑rear grip. They’d also adjust toe‑in for tighter steering response and fine‑tune the suspension geometry to keep the car stable through rapid direction changes.

=== SystemMessage: Pirate ===
[system] You are a pirate. Answer in pirate speak, in at most 2 sentences.
[human] What is a Formula Student car?
[ai] It's a small open-wheel race car designed and built by university students for competition.
[human] Why is the suspension important?
[ai] It keeps the tyres in contact with the road, which controls grip, handling, and driver comfort.
[human] How would a team tune it for a tight autocross track?

[ai response] Ye'd stiffen the front for quick turns, lower the ride height, and tweak the camber to keep the tyres glued to the road. Then ye adjust the rear for a touch of oversteer, so the car can cut corners like a cutlass on a tight autocross track.
```

## 2.2 Messages: ConversationBufferMemory (`2_2_memory.py`)

Implements an in-memory conversation buffer with `ConversationBufferMemory`. On each turn:

1. Load the stored history from the buffer.
2. Send `[SystemMessage, *history, new HumanMessage]` to the model.
3. Save the user input and model output back into the buffer.

The model itself is stateless; the memory object is what carries context. The demo shows the model correctly recalling the user's name and project in later turns.

> **Note:** In LangChain 1.x, `langchain.memory` was moved to the `langchain-classic` package, so the import is `from langchain_classic.memory import ConversationBufferMemory`. It also emits a deprecation warning, which is expected.

**Sample output:**
```
You: Hi, my name is Alex and I'm building a LangChain project.
AI:  Hi Alex! How can I help with your LangChain project?

You: What is my name?
AI:  Alex.

You: What am I building?
AI:  You’re building a LangChain project.

--- Final buffer contents ---
[human] Hi, my name is Alex and I'm building a LangChain project.
[ai] Hi Alex! How can I help with your LangChain project?
[human] What is my name?
[ai] Alex.
[human] What am I building?
[ai] You’re building a LangChain project.
```

## 3.1 PromptTemplate: Few-Shot Sentiment (`3_few_shot_sentiment.py`)

A `FewShotPromptTemplate` with three examples (one positive, one negative, one neutral) and an instruction prefix asking for a one-word answer. It is composed into a chain with `prompt | llm | StrOutputParser()` and run at `temperature=0` for consistent results.

It is tested on six new sentences with known labels, and accuracy is computed as correct predictions divided by total.

**Results:**
```
=== Rendered prompt example ===
Classify the sentiment of the sentence. Answer with exactly one word: positive, negative, or neutral.


Sentence: I absolutely love the new car design, it looks stunning!
Sentiment: positive

Sentence: The engine failed again and the whole day was wasted.
Sentiment: negative

Sentence: The meeting is scheduled for 3 PM tomorrow.
Sentiment: neutral

Sentence: The team's performance at the competition was fantastic!
Sentiment:
===============================

OK  | expected=positive predicted=positive | The team's performance at the competition was fantastic!
OK  | expected=negative predicted=negative | I'm really disappointed with the delayed delivery.
OK  | expected=neutral  predicted=neutral  | The report has 12 pages and 3 appendices.
OK  | expected=negative predicted=negative | This laptop is so slow, I hate using it.
OK  | expected=positive predicted=positive | What a wonderful surprise, thank you so much!
OK  | expected=neutral  predicted=neutral  | The train arrives at platform 4.

Accuracy: 6/6 = 100.0%

Accuracy: <x>/6 = <y>%
```

## 4.1 ToolCalling: Calculator and Wikipedia (`4_tool_calling.py`)

Two custom tools defined with the `@tool` decorator:

- **`calculator`** evaluates a math expression (for example `sqrt(144)`) using a restricted `eval` limited to the `math` module.
- **`wikipedia`** returns a fixed placeholder summary, as the assignment specifies.

The script runs in two stages:

1. **Manual calls:** each tool is invoked directly with `.invoke()` and the results are printed.
2. **Bound to the model:** the tools are attached with `llm.bind_tools([...])`. The model decides whether to call a tool, the script executes each requested call, the results are returned to the model as `ToolMessage`s, and the model produces a final answer.

Test questions: a math question (should call `calculator`), a topic question (should call `wikipedia`), and a greeting (should call no tool).

**Sample output:**
```
=== 1) Manual tool calls ===
calculator('sqrt(144)') -> 12.0
calculator('2**10 + 5') -> 1029
wikipedia('Ada Lovelace') -> [Fixed output] Wikipedia summary for 'Ada Lovelace': This is a placeholder article text.

=== 2) Tools bound to the model ===

Q: What is the square root of 1764?
Tool calls requested: []
A: The square root of 1764 is **42**.

Q: Tell me about Ada Lovelace.
Tool calls requested: [{'name': 'wikipedia', 'args': {'query': 'Ada Lovelace'}, 'id': 'fc_fadef8c7-6963-4603-aa14-de1949e64798', 'type': 'tool_call'}]
Tool result (wikipedia): [Fixed output] Wikipedia summary for 'Ada Lovelace': This is a placeholder article text.
A: **Ada Lovelace (1815 – 1852)**  
Augusta Ada King‑(née Byron) was an English mathematician and writer, best known for her work on Charles Babbage’s early mechanical general‑purpose computer, the Analytical Engine. She is often credited as the world’s first computer programmer.

| Aspect | Details |
|--------|---------|
| **Early life** | Born on 10 December 1815 in London, Ada was the only legitimate child of the poet Lord Byron and his wife, AnnabellaMilbanke. Her mother, a mathematician, encouraged Ada’s interest in numbers and logic from a young age. |
| **Education** | Ada received a rigorous education in mathematics, logic, and foreign languages. She studied under the mathematician Augustus De Morgan and later with the mathematician and philosopher John Herschel. |
| **Analytical Engine** | In 1843, Ada translated an article by Italian mathematician Luigi Menabrea on Babbage’s Analytical Engine. She added extensive notes (the longest single set of notes on the article) that included a description of an algorithm for computing Bernoulli numbers—considered the first published computer program. |
| **Legacy** | Ada’s notes anticipated many concepts of modern computing: conditional branching, loops, and the idea that a machine could manipulate symbols beyond mere numbers. She foresaw that computers could be used for more than arithmetic, envisioning applications in music, graphics, and even artificial intelligence. |
| **Later life** | Ada married William King, 1st Earl of Lovelace, in 1835. She had three children, but her health declined in the 1840s. She died of uterine cancer on 27 November 1852 at the age of 36. |
| **Posthumous recognition** | The U.S. National Institute of Standards and Technology (NIST) named its programming language “Ada” in her honor in 1980. The Ada Lovelace Award, presented by the British Computer Society, recognizes outstanding contributions by women in computing. |
| **Cultural impact** | Ada has become a symbol of women in STEM. Her story is taught in schools worldwide, and she has inspired books,films, and even a NASA mission (the *Ada Lovelace* rover concept). |

**Key Contributions**

1. **First Algorithm** – The algorithm for Bernoulli numbers is widely regarded as the first computer program.  
2. **Vision of General-Purpose Computing** – Ada argued that the Analytical Engine could manipulate symbols and produce music or art, not just perform calculations.  
3. **Notation and Documentation** – Her notes introduced a systematic way to describe machine operations, influencing later computer science documentation.

Ada Lovelace’s blend of mathematical insight, imaginative foresight, and meticulous documentation laid foundational ideas that resonatein modern computing.

Q: Say hello in one short sentence.
Tool calls requested: []
A: Hello!
```

---

## Troubleshooting

- **Model not found or deprecated:** run `python list_models.py` and set `GROQ_MODEL` in `.env` to a currently available chat model ID. Use the exact ID string, not a display name.
- **`ModuleNotFoundError: langchain_classic`:** run `pip install langchain-classic`.
- **Script prints nothing:** check the file is saved and not empty, and that it ends with the `if __name__ == "__main__":` block.
- **`tool_use_failed` from Groq:** rerun the script, or switch to a larger tool-capable model in `.env`.

## Security

Never commit your `.env` file. It is listed in `.gitignore`; confirm with `git status` before pushing.