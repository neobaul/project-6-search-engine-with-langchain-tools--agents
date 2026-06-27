# Import libraries
import os
import sys
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# --- PyQt5 imports ---
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QComboBox, QLabel,
    QGroupBox, QFormLayout, QSplitter
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QTextCursor
# --- Langchain & LangSmith imports ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch as TavilySearchResults
from langchain_core.tools import Tool
# from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
# from langchain_classic import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.embeddings import FakeEmbeddings

# --- Vector Database import ---
from langchain_chroma import Chroma

# Load .env variables
load_dotenv()

import traceback
import sys

def log_exception(exc_type, exc_value, exc_traceback):
    with open("error_log.txt", "w") as f:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    print("CRASH LOG SAVED TO error_log.txt")

sys.excepthook = log_exception

# Explicitly Configuration for LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if not os.getenv("LANGCHAIN_API_KEY"):
    os.environ.get("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "PyQt5_Search_Agent"

# Custom callback handler for LangSmith tracing
class UIThreadCallbackHandler(BaseCallbackHandler):
    """Logs LLM and Agent steps directly to the PyQt Log console."""
    def __init__(self, signal_emitter):
        self.emitter = signal_emitter

    def on_agent_action(self, action, **kwargs):
        self.emitter.emit(f"[Agent Action]: {action.tool} -> Input: {action.tool_input}")

    def on_tool_end(self, output, **kwargs):
        self.emitter.emit(f"[Tool Finished]: Execution yielded result data.\n")

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.emitter.emit(f"[LLM Prompting]: Initiating context window analysis...\n")

# --- Custom Web Search Tool using ---
def scrape_web_page(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        texts = [p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])]
        content = ' '.join(texts).strip()
        return content[:4000]
    except Exception as e:
        return f"An error occurred while scraping: {e}"
    
# --- QThread Background Worker Engine ---
class AgentWorker(QThread):
    """Runs LangChain Execution in a background process to keep PyQt responsive."""
    output_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, prompt, config_data, chat_history):
        super().__init__()
        self.prompt = prompt
        self.config = config_data
        self.chat_history = chat_history

    def get_llm(self):
        provider = self.config['provider']
        model_name = self.config['model']
        api_key = self.config['llm_key']

        if provider == "OpenAI":
            return ChatOpenAI(model=model_name, openai_api_key=api_key, temperature=0.2)
        elif provider == "Groq":
            return ChatGroq(model_name=model_name, groq_api_key=api_key, temperature=0.2)
        elif provider == "Anthropic":
            return ChatAnthropic(model=model_name, anthropic_api_key=api_key, temperature=0.2)
        elif provider == "Ollama":
            return ChatOllama(model=model_name, temperature=0.2)
        raise ValueError("Unsupported LLM provider.")

    def run(self):
        try:
            self.log_signal.emit("[System]: Initializing Tools and LangSmith Tracing Pipeline...\n")

            # Setup Tools
            search_tool = TavilySearchResults(name="TavilySearch", tavily_api_key=self.config['tavily_key'])
            web_scraper_tool = Tool(
                name="WebScraper",
                func=scrape_web_page,
                description="Scrapes full text content down from a given target web URL."
            )
            tools = [search_tool, web_scraper_tool]

            llm = self.get_llm()
            cb_handler = UIThreadCallbackHandler(self.log_signal)

            # Define Routing Architecture
            if self.config['provider'] in ["Groq", "Ollama"]:
                # self.log_signal.emit("[System]: Compiling ReAct Agent Framework...\n")
                # react_prompt = hub.pull("hwchase17/react")
                self.log_signal.emit("[System]: Compiling ReAct Agent Framework...\n")
                from langchain_core.prompts import PromptTemplate
                react_prompt = PromptTemplate.from_template("""You have access to these tools:

                {tools}

                Use this format:
                Question: {input}
                Thought: think about what to do
                Action: tool name, one of [{tool_names}]
                Action Input: input to the tool
                Observation: tool result
                (repeat Thought/Action/Observation as needed)
                Thought: I now know the final answer
                Final Answer: your answer

                Begin!
                Question: {input}
                Thought:{agent_scratchpad}""")
                agent = create_react_agent(llm, tools, react_prompt)

            else:
                self.log_signal.emit("[System]: Compiling Structural Tool-Calling Agent Framework...\n")
                tool_calling_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert desktop engineering researcher application agent."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ])
                agent = create_tool_calling_agent(llm, tools, tool_calling_prompt)

            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

            # Run Agent
            agent_input = {"input": self.prompt, "chat_history": self.chat_history}
            response = agent_executor.invoke(agent_input, {"callbacks": [cb_handler]})

            output_text = response["output"]
            self.output_signal.emit(output_text)

            # Store the resulting data to the Local Vector Database
            self.log_signal.emit("[Database]: Persisting context and response to ChromaDB Vector Database...\n")

            # Using OpenAI embeddings if available, fallback to a dummy embedding function for structure compatibility
            if self.config['provider'] == "OpenAI":
                embeddings = OpenAIEmbeddings(openai_api_key=self.config['llm_key'])
            else:
                embeddings = FakeEmbeddings(size=1536)

            # Initialize a persistent vector store directory
            db = Chroma.from_texts(
                texts=[f"Query: {self.prompt} | Answer: {output_text}"],
                embedding=embeddings,
                persist_directory="./chroma_db_storage"
            )
            self.log_signal.emit("[Database]: Successfully Saved Vector Encodings.\n")

        except Exception as e:
            self.log_signal.emit(f"[Execution Error]: {str(e)}\n")
            self.output_signal.emit(f"An unexpected failure occurred: {str(e)}")
        finally:
            self.finished_signal.emit()


# --- Main Window Client Interface Design ---
class HybridResearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔮 Advanced AI Search Engine (PyQt5 + LangChain)")
        self.resize(1200, 800)
        self.chat_history = []
        self.init_ui()
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Column: Configuration Settings Panel
        config_box = QGroupBox("Configuration Engine")
        config_layout = QFormLayout(config_box)

        self.tavily_key_input = QLineEdit()
        self.tavily_key_input.setEchoMode(QLineEdit.Password)
        self.tavily_key_input.setText(os.getenv("TAVILY_API_KEY", ""))
        config_layout.addRow(QLabel("Tavily API Key:"), self.tavily_key_input)

        self.provider_select = QComboBox()
        self.provider_select.addItems(["Groq", "OpenAI", "Anthropic", "Ollama"])
        self.provider_select.currentTextChanged.connect(self.update_models)
        config_layout.addRow(QLabel("LLM Provider:"), self.provider_select)

        self.model_select = QComboBox()
        config_layout.addRow(QLabel("Model Selection:"), self.model_select)

        self.llm_key_input = QLineEdit()
        self.llm_key_input.setEchoMode(QLineEdit.Password)
        config_layout.addRow(QLabel("LLM API Key:"), self.llm_key_input)

        self.update_models(self.provider_select.currentText())
        config_box.setFixedWidth(320)
        main_layout.addWidget(config_box)

        # Right Column: Main split screen space for interactive Chat and Trace Logging logs
        splitter = QSplitter(Qt.Vertical)

        # Chat interface
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        chat_layout.addWidget(QLabel("Interactive Chat Console:"))
        chat_layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask a comprehensive research question...")
        self.query_input.returnPressed.connect(self.execute_search)
        self.send_btn = QPushButton("Execute Search")
        self.send_btn.clicked.connect(self.execute_search)

        input_layout.addWidget(self.query_input)
        input_layout.addWidget(self.send_btn)
        chat_layout.addLayout(input_layout)

        splitter.addWidget(chat_widget)

        # Real-time System Execution Trace logs console
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("background-color: #1e1e1e; color: #a9dc76; font-family: Courier New;")
        log_layout.addWidget(QLabel("Live System Traces & Execution Logs (Connected to LangSmith):"))
        log_layout.addWidget(self.log_display)

        splitter.addWidget(log_widget)
        main_layout.addWidget(splitter)

    def update_models(self, provider):
        self.model_select.clear()
        models = {
            "OpenAI": ["gpt-4o", "gpt-3.5-turbo"],
            "Groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
            "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
            "Ollama": ["llama3:8b", "qwen3:0.6b"]
        }
        self.model_select.addItems(models.get(provider, []))

        if provider == "Ollama":
            self.llm_key_input.setDisabled(True)
            self.llm_key_input.setPlaceholderText("Local deployment (No key required)")
        else:
            self.llm_key_input.setEnabled(True)
            self.llm_key_input.setPlaceholderText(f"Enter {provider} API Key")
            self.llm_key_input.setText(os.getenv(f"{provider.upper()}_API_KEY", ""))

    def append_chat(self, sender, message):
        self.chat_display.append(f"<b>{sender}:</b> {message}<br>")
        self.chat_display.moveCursor(QTextCursor.End)

    def append_log(self, log_msg):
        self.log_display.insertPlainText(log_msg)
        self.log_display.moveCursor(QTextCursor.End)

    def execute_search(self):
        prompt = self.query_input.text().strip()
        if not prompt:
            return

        tavily_key = self.tavily_key_input.text().strip()
        provider = self.provider_select.currentText()
        llm_key = self.llm_key_input.text().strip()

        if not tavily_key:
            self.append_chat("System", "Error: Missing Tavily Search API key Configuration.")
            return          
        if provider != "Ollama" and not llm_key:
            self.append_chat("System", f"Error: Missing API key context for {provider}.")
            return

        self.append_chat("User", prompt)
        self.query_input.clear()
        self.send_btn.setEnabled(False)

        # Package target configurations
        config_data = {
            'provider': provider,
            'model': self.model_select.currentText(),
            'tavily_key': tavily_key,
            'llm_key': llm_key
        }

        # Initialize the background processing Thread
        self.worker = AgentWorker(prompt, config_data, self.chat_history)
        self.worker.output_signal.connect(lambda text: self.append_chat("Assistant", text))
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(lambda: self.send_btn.setEnabled(True))

        # Append parameters to local history memory maps
        self.chat_history.append(HumanMessage(content=prompt))
        self.worker.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    client = HybridResearchApp()
    client.show()
    sys.exit(app.exec_())