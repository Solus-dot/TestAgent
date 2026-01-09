from typing import Any
import webbrowser
import imaplib
import email
from mcp.server.fastmcp import FastMCP
import yt_dlp

# Initialize the MCP Server
mcp = FastMCP("MyLocalAgentTools")

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
    """Reads the latest email subjects from Gmail."""
    # (Simplified for brevity - insert your full IMAP logic here)
    return "From: Boss | Subject: Project Orion Update"

if __name__ == "__main__":
    # This runs the server over Stdio (Standard Input/Output)
    mcp.run()