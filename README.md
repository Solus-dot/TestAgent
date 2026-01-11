# TestAgent
Python-based AI Agent for tool-calls.

## Overview
Agent uses a lmstudio-cli server for inference and Model Context Protocol (MCP) for basic tool calls.\
Small project to test agentic capabilities of small language models and versatility of MCP.\
Mostly vibecoded AI slop, but not bad for something as small as this.

## Install and Run:
### Prerequisites
* **Python 3.11+** (`python3 --version`)
* **uv** (Modern Python package manager). [Install uv](https://github.com/astral-sh/uv)
* **LM Studio** (With CLI enabled). Run `lms bootstrap` in your terminal if you haven't already.

### 1. Installation
Clone the repo and set up the environment in one command using `uv`:

```bash
# Clone the repository
git clone https://github.com/Solus-dot/TestAgent.git
cd TestAgent

# Install dependencies (This creates the .venv automatically)
uv sync --python 3.11.14
```
### 2. Configuration
Create a .env file in the root directory to enable the tools:

```bash
nano .env
```
#### Required Variables:

```ini
# Email Tool Configuration
EMAIL_USER="your_email@gmail.com"
EMAIL_PASS="your_app_password"  # Generated 16-char App Password (NOT your login password)
IMAP_SERVER="imap.gmail.com"    # or imap-mail.outlook.com
```

### 3. Running the Agent
You need two terminal windows open.

**Terminal 1 - The Model Server:** This script starts the LM Studio server and lets you select a model.

```bash
./llmServer.sh
# Follow the prompts to select a model (e.g., Llama 3, Qwen)
```

**Terminal 2 - The Agent Client:** This runs your Python agent which connects to the server and handles tools.

```bash
uv run src/main.py
```