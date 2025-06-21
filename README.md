# Medical Diagnostics AI System

## Overview

This project is an advanced AI-powered medical diagnostics system that leverages multi-agent orchestration, LLMs (Gemini, Groq), and vector search (Qdrant) to analyze medical text and documents. It provides detailed, professional medical reports and supports interactive chat for follow-up questions, all via a beautiful terminal interface using Rich.

## Features
- Multi-specialty agent orchestration (cardiology, neurology, etc.)
- Document and text input (PDF, DOCX, TXT, CSV, XLSX, JPG, PNG)
- Professional, detailed medical reports
- Interactive chat with context-aware answers
- Qdrant vector search for enhanced context
- Local SQLite storage (no cloud, no Docker required)
- Robust error handling and fallback analysis
- All prompts externalized to `prompts.yaml`
- Modular, maintainable codebase

## Setup
1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd AI-Agents-for-Medical-Diagnostics
   ```
2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Configure environment:**
   - Copy `.env.example` to `.env` and fill in your API keys (Gemini, Groq, etc.)
   - Edit `config.yaml` as needed for model and database settings.

5. **(Optional) Start Qdrant locally:**
   - Download and run Qdrant from https://qdrant.tech/

## Usage
Run the CLI:
```sh
python main.py
```
Follow the on-screen menu to analyze text, upload documents, or chat about your results.

## Directory Structure
```
medical_analysis/   # Main package (agents, utils, extractors)
data/               # Input/sample data (not tracked in git)
outputs/            # Generated reports and outputs (not tracked in git)
logs/               # Log files (not tracked in git)
results/            # Analysis results (if used)
prompts.yaml        # All LLM and agent prompts
requirements.txt    # Python dependencies
config.yaml         # System configuration
.env                # API keys and secrets (not tracked in git)
main.py             # Entry point CLI
```

## License
See [LICENSE](LICENSE).

## Streamlit Web Frontend

A visually stunning, animated chatbot interface is available via Streamlit.

### Features
- Chatbot interface for medical Q&A and report analysis
- Animated transitions and loading indicators
- Context-aware, multi-agent answers
- Beautiful, modern UI with medical theme
- Handles document upload, text input, and chat history

### Running the Web App

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Open your browser to the provided local URL. 