from typing import Any
import webbrowser
import imaplib
import email
import requests
import psutil
from dotenv import load_dotenv
import os
from mcp.server.fastmcp import FastMCP
import yt_dlp

# Initialize the MCP Server
mcp = FastMCP("MyLocalAgentTools")

# Extract .env variables
load_dotenv()
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
IMAP_SERVER = os.getenv('IMAP_SERVER')

# --- Define Tools ---

@mcp.tool()
def play_youtube(topic: str) -> str:
    """Searches YouTube for the topic and plays the first video found."""

    print(f"  > [Tool] Searching YouTube for: {topic}...")

    # Define a fallback URL (Search Results) in case the direct play fails
    fallback_url = f"https://www.youtube.com/results?search_query={topic.replace(' ', '+')}"
    
    try:
        # Use yt-dlp to search YouTube
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'default_search': 'ytsearch1'  # Search for 1 result
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
            
            # Open the video URL
            print(f"  > [Tool] Playing: {title}")
            webbrowser.open(video_url)
            
            return f"Success: Now playing '{title}' in the browser."
    
    except Exception as e:
        # Fallback: If anything crashes, open the search page
        print(f"  > [Tool Error] {e}")
        webbrowser.open(fallback_url)
        return f"I encountered an error trying to play the video directly ({str(e)}), so I opened the search results instead."


@mcp.tool()
def read_latest_email(count: int = 1) -> str:
    """Fetches the latest N emails from your inbox. Returns sender, subject, and a preview of the body text."""
    print(f"  > [Tool] Fetching last {count} email(s)...")
    
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
                    f"📩 FROM: {sender}\n"
                    f"SUBJECT: {subject}\n"
                    f"BODY: {body_preview}"
                )
        
        mail.logout()
        return "\n\n".join(results) if results else "No emails found."
    
    except imaplib.IMAP4.error as e:
        return f"IMAP error: {e}. Check your email/password and IMAP settings."
    except Exception as e:
        print(f"  > [Tool Error] {e}")
        return f"Unexpected error: {e}"


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a specific city using wttr.in."""
    print(f"  > [Tool] Checking weather for: {city}...")
    try:
        # format=3 gives a one-line output like: "Paris: ⛅️ +12°C"
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.text.strip()
        else:
            return "Could not fetch weather data."
    except Exception as e:
        print(f"  > [Tool Error] {e}")
        return f"Error connecting to weather service: {e}"
    

@mcp.tool()
def get_system_stats() -> str:
    """Checks the current CPU and RAM usage of the computer."""
    print(f"  > [Tool] Fetching CPU and RAM stats...")
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Convert bytes to GB
        total_mem_gb = memory.total / (1024 ** 3)
        used_mem_gb = memory.used / (1024 ** 3)
        
        return (f"CPU Usage: {cpu_percent}%\n"
                f"RAM Usage: {used_mem_gb:.1f}GB / {total_mem_gb:.1f}GB ({memory.percent}%)")
    except Exception as e:
        print(f"  > [Tool Error] {e}")
        return f"Unexpected error: {e}"


if __name__ == "__main__":
    # This runs the server over Stdio (Standard Input/Output)
    mcp.run()