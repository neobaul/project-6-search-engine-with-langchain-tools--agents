# AI Search Engine — PyQt5 + LangChain Tools & Agents

A desktop AI research assistant built with PyQt5 and LangChain. Type a question, and an AI agent searches the web via Tavily, scrapes pages for deeper content, reasons through the answer, and responds — all while displaying its thought process in a live log console. Supports multiple LLM providers and stores every Q&A pair in a local ChromaDB vector database.

---

## What It Does

- **Web search** via Tavily API — finds the most relevant pages for your query
- **Deep web scraping** — a custom tool dives into individual pages to extract full text content
- **Dual agent routing** — uses ReAct agents for Groq/Ollama, Tool-Calling agents for OpenAI/Anthropic
- **Live execution traces** — a green-on-black log console shows every agent action and LLM call in real time
- **LangSmith tracing** — all agent steps are sent to your LangSmith dashboard for inspection
- **Vector memory** — every question + answer is embedded and stored locally in ChromaDB
- **Multi-turn chat** — conversation history is passed to the agent on each turn

---

## Project Structure

```
├── advanced_research_project.py   # Main PyQt5 desktop application
├── synopsis of the project.txt    # Detailed walkthrough of the code
└── .gitignore
```

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/neobaul/project-6-search-engine-with-langchain-tools--agents.git
cd project-6-search-engine-with-langchain-tools--agents
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install pyqt5 python-dotenv requests beautifulsoup4
pip install langchain-openai langchain-groq langchain-anthropic langchain-ollama
pip install langchain-tavily langchain-chroma langchain-classic chromadb
```

### 4. Set up API keys
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_key_here     # optional, for LangSmith tracing
```
Get a free Tavily API key at [tavily.com](https://tavily.com).

### 5. Run the app
```bash
python advanced_research_project.py
```

---

## How It Works

```
You type a question
    ↓
AgentWorker spins up in a background QThread
    ↓
Agent receives two tools:
  - TavilySearch  → finds relevant URLs
  - WebScraper    → extracts full text from a URL
    ↓
Agent routing:
  - Groq / Ollama  → ReAct agent (Thought → Action → Observation loop)
  - OpenAI / Anthropic → Tool-Calling agent (native structured tool use)
    ↓
UIThreadCallbackHandler logs every step to the green console
    ↓
Final answer sent to chat panel
    ↓
Q&A pair embedded and stored in ChromaDB (./chroma_db_storage)
```

The UI stays fully responsive throughout — all agent work runs in a background thread, communicating back via PyQt5 signals.

---

## UI Layout

**Left panel — Configuration Engine**
- Tavily API key
- LLM provider selector (Groq, OpenAI, Anthropic, Ollama)
- Model selector (updates automatically per provider)
- LLM API key (disabled for Ollama — no key needed for local models)

**Top right — Chat Console**
- Conversation history
- Query input + Execute button

**Bottom right — Live Trace Console**
- Real-time agent execution logs
- Shows tool calls, LLM prompts, and tool results as they happen

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Desktop UI | PyQt5 |
| Web Search | Tavily API |
| Web Scraping | Requests + BeautifulSoup4 |
| LLM Providers | OpenAI, Groq, Anthropic, Ollama |
| Agent Frameworks | ReAct (open-source models), Tool-Calling (OpenAI/Anthropic) |
| Orchestration | LangChain LCEL + AgentExecutor |
| Vector Storage | ChromaDB (local persistent) |
| Tracing | LangSmith |
| Background Tasks | PyQt5 QThread + pyqtSignal |
| Language | Python 3.11 |
