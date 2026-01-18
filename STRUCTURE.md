# Repository Structure

Your repository has been reorganized for better clarity and maintainability. Here's the new structure:

```
TestAgent/
├── main.py                 # Entry point - Run this to start the agent
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── uv.lock               # Dependency lock file
│
├── agent/                 # Agent logic
│   ├── __init__.py
│   └── main.py           # Main agent implementation with MCP integration
│
├── core/                  # Core functionality
│   ├── __init__.py
│   ├── memory.py         # Vector memory system for long-term storage
│   └── prompts.py        # System prompts and agent instructions
│
├── tools/                 # MCP Tools server
│   ├── __init__.py
│   └── tools_server.py   # Tools definitions (email, weather, system stats, etc.)
│
├── scripts/               # Utility scripts
│   └── llmServer.sh      # LM Studio server launcher and log formatter
│
├── src/utils/             # Utility modules
│   ├── __init__.py
│   └── format_logs.py    # Log formatting for LM Studio output
│
└── memories/             # Agent memory storage
    ├── agent_memory.pkl  # Serialized vector memory database
    └── memories_export.txt # Human-readable memory export
```

## Quick Start

1. **Run the agent:**
   ```bash
   uv run main.py
   ```

2. **Start the LM Studio server:**
   ```bash
   bash scripts/llmServer.sh
   ```

## Module Descriptions

- **`agent/`** - Contains the main agent logic, chat loop, and MCP client setup
- **`core/`** - Core abstractions like memory management and system prompts
- **`tools/`** - MCP server defining all available tools (memory, email, weather, system stats)
- **`scripts/`** - Helper scripts for development and infrastructure
- **`src/utils/`** - Shared utilities for logging and formatting

## Key Changes from Previous Structure

✅ Removed cluttered `src/` directory  
✅ Organized modules by functionality (agent, core, tools)  
✅ Added root `main.py` entry point  
✅ Scripts in dedicated `scripts/` folder  
✅ Much easier to navigate and find components  
