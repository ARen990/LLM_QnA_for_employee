# Agentic AI: Multi-Agent RAG System

## Objective
This repository contains the implementation of a simple Agentic AI system using multi-agent orchestration and Retrieval-Augmented Generation (RAG). It was developed as part of the AI Engineer Programming Test to demonstrate skills in LLM integration, prompt design, and agent coordination.

## System Architecture
The system employs a sequential workflow orchestrated via **LangGraph**, consisting of two specialized agents:
1. **Data Retriever Agent:** Extracts core keywords from the user's query and performs a keyword-based search against a local knowledge base (`knowledge_base.txt`). It returns raw, relevant text snippets.
2. **Report Generator Agent:** Synthesizes the retrieved snippets to generate a comprehensive, non-redundant, and well-formatted response for the user. It strictly avoids hallucination by relying solely on the provided context.

## Tech Stack
* **Language:** Python 3
* **Framework:** Langchain, Langgraph
* **LLM:** Google Gemini API (gemini-3.5-flash)
* **Environment Management:** python-dotenv

## Repository Structure
* `main.py` - Core logic for multi-agent orchestration (Sequential Workflow).
* `interactive_test.py` - An interactive CLI version for real-time user queries.
* `knowledge_base.txt` - The simulated knowledge base containing company policies and guidelines.
* `requirements.txt` - Project dependencies.
* `/screenshots` - Directory containing execution results for various sample queries.

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
   cd your-repo-name
   ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Configure Environment Variables:
Create a .env file in the root directory and add your API key:
    ```bash
    GOOGLE_API_KEY="your_gemini_api_key_here"
    ```
## Usage
To run the pre-defined test queries, execute:
  ```bash
  python main.py
  ```
To run the system in an interactive loop (CLI), execute:
  ```bash
  python interactive_test.py
  ```
    
