from typing import Any
import webbrowser
import imaplib
import email
import requests
import psutil
from dotenv import load_dotenv
import os
import sys
from mcp.server.fastmcp import FastMCP
import yt_dlp

# Add parent directory to path to import memory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.memory import VectorMemory

# Initialize the MCP Server
mcp = FastMCP("MyLocalAgentTools")

# Extract .env variables
load_dotenv()
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
IMAP_SERVER = os.getenv('IMAP_SERVER')

# Initialize memory system directly in the tools server
MEMORY_PATH = os.getenv('MEMORY_PATH', 'memories/agent_memory.pkl')
memory_system = VectorMemory(storage_path=MEMORY_PATH)

print(f"[TOOLS SERVER] Memory system initialized with {len(memory_system.memories)} memories", file=sys.stderr)

# --- Memory Tools ---

@mcp.tool()
def store_memory(text: str, memory_type: str = "fact", importance: str = "normal") -> str:
    """
    Store information in long-term memory for future recall.
    Use this when the user shares important information about themselves, their preferences,
    or asks you to remember something specific.
    
    Args:
        text: The information to remember (be specific and clear)
        memory_type: Type of memory - "fact" (general info), "preference" (likes/dislikes),
                     "identity" (name, job, personal details), "goal" (user's objectives)
        importance: How important is this - "low", "normal", "high"
    
    Returns:
        Confirmation message
    
    Examples:
        - User says "My name is Alex" -> store_memory("User's name is Solus", "identity", "high")
        - User says "I love pizza" -> store_memory("User loves pizza", "preference", "normal")
        - User says "Remember I'm learning Python" -> store_memory("User is learning Python", "goal", "normal")
    """
    try:
        metadata = {
            "importance": importance,
            "stored_by": "llm"
        }
        
        memory_system.add(text, metadata=metadata, memory_type=memory_type)
        return f"[MEMORY TOOL] Stored: '{text[:50]}...' ({memory_type}, {importance})"
    except Exception as e:
        return f"[MEMORY TOOL ERROR] Failed to store: {e}"
    
@mcp.tool()
def recall_memory(query: str, memory_type: str = "", top_k: int = 3) -> str:
    """
    Search long-term memory for relevant information.
    Use this when you need context about the user's previous conversations, preferences, or facts.
    
    Args:
        query: What to search for (be specific about what information you need)
        memory_type: Optional filter - "fact", "preference", "identity", "goal", or "" for all types
        top_k: How many relevant memories to retrieve (1-5)
    
    Returns:
        Relevant memories with relevance scores
    
    Examples:
        - User asks "What's my name?" -> recall_memory("user's name", "identity", 1)
        - User asks "What do I like?" -> recall_memory("user likes preferences", "preference", 3)
        - User asks about past conversation -> recall_memory("previous topic", "", 5)
    """
    try:
        # Reload from disk to get latest memories
        memory_system.load()
        
        mem_type = memory_type if memory_type else None        
        results = memory_system.search(query, top_k=top_k, memory_type=mem_type, min_score=0.3)
        
        if not results:
            return f"[MEMORY TOOL] No relevant memories found for '{query}'"
        
        output = f"[MEMORY TOOL] Found {len(results)} relevant memories:\n\n"
        for i, mem in enumerate(results, 1):
            output += f"{i}. [ID: {mem['id']}] [{mem['type']}] {mem['text']} (score: {mem['score']:.2f})\n"
        
        return output
    except Exception as e:
        return f"[MEMORY TOOL ERROR] Failed to recall: {e}"
    
@mcp.tool()
def list_recent_memories(count: int = 5, memory_type: str = "") -> str:
    """
    List the most recent memories stored.
    
    Args:
        count: How many recent memories to show (1-10)
        memory_type: Optional filter by type (or "" for all types)
    
    Returns:
        List of recent memories
    """
    try:
        # Reload from disk to get latest memories
        memory_system.load()
        
        memories = memory_system.memories
        
        if memory_type:
            memories = [m for m in memories if m["type"] == memory_type]
        
        if not memories:
            return f"[MEMORY TOOL] No memories found" + (f" of type '{memory_type}'" if memory_type else "")
        
        # Sort by timestamp (newest first)
        sorted_memories = sorted(memories, key=lambda x: x["timestamp"], reverse=True)[:count]
        
        output = f"📝 {len(sorted_memories)} most recent memories:\n\n"
        for i, mem in enumerate(sorted_memories, 1):
            import time
            age = (time.time() - mem["timestamp"]) / 60  # minutes
            if age < 60:
                age_str = f"{age:.0f}m ago"
            else:
                age_str = f"{age/60:.1f}h ago"
            
            output += f"{i}. [ID: {mem['id']}] [{mem['type']}] {mem['text']} ({age_str})\n"
        
        return output
    except Exception as e:
        return f"[MEMORY TOOL ERROR] Failed to list: {e}"

@mcp.tool()
def forget_memory(memory_id: int) -> str:
    """
    Delete a specific memory by ID.
    Use when user asks to forget something or if information is outdated.
    
    Args:
        memory_id: The ID of the memory to delete (get this from list_recent_memories)
    
    Returns:
        Confirmation message
    """
    try:
        # Reload from disk to get latest memories
        memory_system.load()
        
        success = memory_system.delete(memory_id)
        if success:
            return f"[MEMORY TOOL] Deleted memory #{memory_id}"
        else:
            return f"[MEMORY TOOL] Memory #{memory_id} not found"
    except Exception as e:
        return f"[MEMORY TOOL] Failed to delete: {e}"

# --- Other Tools ---

@mcp.tool()
def play_youtube(topic: str) -> str:
    """Searches YouTube for the topic and plays the first video found."""
    print(f"[TOOL] Searching YouTube for: {topic}...", file=sys.stderr)
    fallback_url = f"https://www.youtube.com/results?search_query={topic.replace(' ', '+')}"
    
    try:
        # Use yt-dlp to search YouTube
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'default_search': 'ytsearch1'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch1:{topic}", download=False)
            
            if not result or 'entries' not in result or not result['entries']:
                # Fallback: No video found
                webbrowser.open(fallback_url)
                return f"Could not find a specific video link, so I opened the search results for '{topic}'."
            
            # Extract first video
            video = result['entries'][0]
            title = video.get('title', 'Unknown Title')
            video_url = video.get('url') or f"https://www.youtube.com/watch?v={video['id']}"
            
            print(f"[TOOL] Playing: {title}", file=sys.stderr)
            webbrowser.open(video_url)
            
            return f"Success: Now playing '{title}' in the browser."
    
    except Exception as e:
        print(f"[TOOL ERROR] {e}", file=sys.stderr)
        webbrowser.open(fallback_url)
        return f"I encountered an error trying to play the video directly ({str(e)}), so I opened the search results instead."


@mcp.tool()
def read_latest_email(count: int = 1) -> str:
    """Fetches the latest N emails from your inbox. Returns sender, subject, and a preview of the body text."""
    print(f"[TOOL] Fetching last {count} email(s)...", file=sys.stderr)
    
    try:
        # Connect and authenticate
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")
        
        # Search for all messages (or use "UNSEEN" for unread only)
        status, messages = mail.search(None, "ALL")
        if status != "OK":
            return "Failed to search inbox."
        
        email_ids = messages[0].split()
        if not email_ids:
            return "Your inbox is empty."
        
        # Grab the most recent N emails
        latest_ids = email_ids[-count:]
        results = []
        
        # Process each email (newest first)
        for e_id in reversed(latest_ids):
            _, msg_data = mail.fetch(e_id, "(RFC822)")
            
            for response_part in msg_data:
                if not isinstance(response_part, tuple):
                    continue
                
                msg = email.message_from_bytes(response_part[1])
                
                # Decode the subject line
                subject_header = email.header.decode_header(msg["Subject"])[0]
                subject = subject_header[0]
                if isinstance(subject, bytes):
                    encoding = subject_header[1] or "utf-8"
                    subject = subject.decode(encoding, errors="ignore")
                
                sender = msg.get("From", "Unknown Sender")
                
                # Extract plain text body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True)
                            if payload:
                                body = payload.decode(errors="ignore")
                                break
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        body = payload.decode(errors="ignore")
                
                # Trim body to avoid overwhelming the LLM with huge emails
                body_preview = body.strip()[:500]
                if len(body.strip()) > 500:
                    body_preview += "..."
                
                results.append(
                    f"FROM: {sender}\n"
                    f"SUBJECT: {subject}\n"
                    f"BODY: {body_preview}"
                )
        
        mail.logout()
        return "\n\n".join(results) if results else "No emails found."
    
    except imaplib.IMAP4.error as e:
        return f"IMAP error: {e}. Check your email/password and IMAP settings."
    except Exception as e:
        print(f"[TOOL ERROR] {e}", file=sys.stderr)
        return f"Unexpected error: {e}"


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a specific city using wttr.in."""
    print(f"[TOOL] Checking weather for: {city}...", file=sys.stderr)
    try:
        # format=3 gives a one-line output like: "Paris: ⛅️ +12°C"
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.text.strip()
        else:
            return "Could not fetch weather data."
    except Exception as e:
        print(f"[TOOL ERROR] {e}", file=sys.stderr)
        return f"Error connecting to weather service: {e}"
    

@mcp.tool()
def get_system_stats() -> str:
    """Checks the current CPU and RAM usage of the computer."""
    print(f"[TOOL] Fetching CPU and RAM stats...", file=sys.stderr)
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Convert bytes to GB
        total_mem_gb = memory.total / (1024 ** 3)
        used_mem_gb = memory.used / (1024 ** 3)
        
        return (f"CPU Usage: {cpu_percent}%\n"
                f"RAM Usage: {used_mem_gb:.1f}GB / {total_mem_gb:.1f}GB ({memory.percent}%)")
    except Exception as e:
        print(f"[TOOL ERROR] {e}", file=sys.stderr)
        return f"Unexpected error: {e}"


if __name__ == "__main__":
    # This runs the server over Stdio (Standard Input/Output)
    mcp.run()
