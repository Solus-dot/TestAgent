from typing import Any
import webbrowser
import imaplib
import email
from mcp.server.fastmcp import FastMCP

# Initialize the MCP Server
mcp = FastMCP("MyLocalAgentTools")

# --- Define Tools ---

@mcp.tool()
def play_youtube(topic: str) -> str:
    """Opens a YouTube search for the topic."""
    url = f"https://www.youtube.com/results?search_query={topic.replace(' ', '+')}"
    webbrowser.open(url)
    return f"Opened YouTube for {topic}"

@mcp.tool()
def read_latest_email(count: int = 1) -> str:
    """Reads the latest email subjects from Gmail."""
    # (Simplified for brevity - insert your full IMAP logic here)
    return "From: Boss | Subject: Project Orion Update"

if __name__ == "__main__":
    # This runs the server over Stdio (Standard Input/Output)
    mcp.run()