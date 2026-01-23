import asyncio
import os
import re
import sys
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set environment variable for memory path (so tools_server.py uses same file)
MEMORY_PATH = 'memories/agent_memory.pkl'
os.environ['MEMORY_PATH'] = MEMORY_PATH

# Import after setting env var
from core.memory import VectorMemory
from core.prompts import SYSTEM_PROMPT

# Connect to the Llama Server
llm_client = OpenAI(base_url="http://127.0.0.1:6767/v1", api_key="TestAgent")

# Define connection to the MCP Server
script_dir = os.path.dirname(os.path.abspath(__file__))
server_params = StdioServerParameters(
    command="python3", 
    args=[os.path.join(os.path.dirname(script_dir), "tools", "tools_server.py")],
    env={**os.environ, 'MEMORY_PATH': MEMORY_PATH}
)

# Initialize memory system
memory = VectorMemory(storage_path=MEMORY_PATH)

def extract_final_response(text):
    """Extract only the final user-facing response from structured output"""
    if not text:
        return ""
    
    # Remove <think>...</think> blocks (Nemotron reasoning)
    text = re.sub(r'^(.*?)\</think>', '', text, flags=re.DOTALL)
    
    # Try structured format (gpt-oss, nemotron with channels)
    final_pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)'
    final_match = re.search(final_pattern, text, re.DOTALL)
    
    if final_match:
        return final_match.group(1).strip()
    
    # Try alternative message format (some nemotron configs)
    message_pattern = r'<\|start\|>assistant.*?<\|message\|>(.*?)(?:<\|end\|>|$)'
    message_matches = re.findall(message_pattern, text, re.DOTALL)
    
    if message_matches:
        return message_matches[-1].strip()
    
    # Try simple assistant tag (fallback for standard templates)
    simple_pattern = r'<\|assistant\|>(.*?)(?:<\||\Z)'
    simple_match = re.search(simple_pattern, text, re.DOTALL)
    
    if simple_match:
        return simple_match.group(1).strip()
    
    # No special tokens found - return as-is
    if '<|' not in text and '<think>' not in text:
        return text.strip()
    
    # Fallback: aggressively remove all special tokens
    cleaned = text
    cleaned = re.sub(r'<\|start\|>.*?<\|message\|>', '', cleaned)
    cleaned = re.sub(r'<\|end\|>', '', cleaned)
    cleaned = re.sub(r'<\|channel\|>\w+', '', cleaned)
    cleaned = re.sub(r'<\|constrain\|>\w+', '', cleaned)
    cleaned = re.sub(r'to=functions\.\w+', '', cleaned)
    cleaned = re.sub(r'<\|.*?\|>', '', cleaned)
    
    return cleaned.strip()
async def run_agent():
    print("--- MCP Agent Connecting... ---")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            await session.initialize()
            mcp_tools_list = await session.list_tools()
            
            # Convert MCP tools format to OpenAI tools format
            openai_tools = []
            for tool in mcp_tools_list.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            
            print(f"Loaded {len(openai_tools)} tools from MCP server.")
            
            # The Chat Loop
            history = []
            
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ["quit", "exit"]: 
                    break
                
                # Special memory commands
                if user_input.lower().startswith("/memory"):
                    handle_memory_command(user_input)
                    continue
                
                # Add user message to history first
                history.append({"role": "user", "content": user_input})
                
                # Build messages for API call
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages.extend(history)

                # Ask LLM
                response = llm_client.chat.completions.create(
                    model="TestAgent-model",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    extra_body={
                        "enable_thinking": False
                    }
                )

                msg = response.choices[0].message
                
                # Loop to handle tool call chaining
                while msg.tool_calls:
                    num_calls = len(msg.tool_calls)
                    if num_calls > 1:
                        print(f"[AGENT] {num_calls} parallel tool calls:")
                    else:
                        print(f"[AGENT] Tool call:")
                    
                    assistant_content = extract_final_response(msg.content or "")

                    # Create assistant message object
                    assistant_msg = {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in msg.tool_calls
                        ]
                    }
                    
                    # Add to history
                    history.append(assistant_msg)
                    
                    # Execute ALL tool calls
                    for i, tool_call in enumerate(msg.tool_calls, 1):
                        fname = tool_call.function.name
                        fargs = tool_call.function.arguments
                        
                        # Show tool call with index if multiple
                        if num_calls > 1:
                            print(f"[AGENT] [{i}/{num_calls}] {fname}({fargs})")
                        else:
                            print(f"[AGENT] {fname}({fargs})")

                        # Execute via MCP Protocol
                        import json
                        args_dict = json.loads(fargs)
                        
                        result = await session.call_tool(fname, arguments=args_dict)
                        tool_output = result.content[0].text
                        
                        # Show abbreviated result
                        if num_calls > 1:
                            preview = tool_output[:80] + "..." if len(tool_output) > 80 else tool_output
                            print(f"[AGENT] {preview}")
                        else:
                            preview = tool_output[:150] + "..." if len(tool_output) > 150 else tool_output
                            print(f"[AGENT] Result: {preview}")

                        # Create tool message object
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output
                        }
                        
                        # Add to history
                        history.append(tool_msg)
                    
                    if num_calls > 1:
                        print(f"[AGENT] Synthesizing response from {num_calls} results...")
                    
                    # Rebuild messages with updated history
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    messages.extend(history)
                    
                    # Get next response
                    response = llm_client.chat.completions.create(
                        model="TestAgent-model",
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto",
                        extra_body={
                            "enable_thinking": False
                        }
                    )
                    msg = response.choices[0].message
                    
                # Final Text Response (no more tool calls)
                raw_response = msg.content
                assistant_response = extract_final_response(raw_response)
                
                print(f"Agent: {assistant_response}")
                history.append({"role": "assistant", "content": assistant_response})

def handle_memory_command(command: str):
    """Handle special memory commands"""
    # Reload memory to get latest from disk
    memory.load()
    
    parts = command.split()
    
    if len(parts) == 1 or parts[1] == "stats":
        stats = memory.get_stats()
        print("\n[AGENT] Memory Statistics:")
        print(f"Total memories: {stats['total_memories']}")
        if stats['total_memories'] > 0:
            print(f"Oldest: {stats['oldest']['text']} ({stats['oldest']['age_days']:.1f} days old)")
            print(f"Newest: {stats['newest']['text']} ({stats['newest']['age_days']:.1f} days old)")
            print(f"Most accessed: {stats['most_accessed']['text']} ({stats['most_accessed']['count']} times)")
            print(f"Types: {stats['types']}")
    
    elif parts[1] == "search" and len(parts) > 2:
        query = " ".join(parts[2:])
        results = memory.search(query, top_k=5)
        print(f"\n[AGENT] Search results for '{query}':")
        if results:
            for r in results:
                print(f"  [{r['score']:.2f}] {r['text']}")
        else:
            print(f"  No results found")
    
    elif parts[1] == "export":
        memory.export_txt()
        print("[AGENT] Exported to memories/memories_export.txt")
    
    elif parts[1] == "clear":
        confirm = input("[AGENT] Are you sure you want to clear ALL memories? (yes/no): ")
        if confirm.lower() == "yes":
            memory.clear()
            print("[AGENT] All memories cleared")
        else:
            print("[AGENT] Cancelled")
    
    else:
        print("\nMemory commands:")
        print("  /memory stats          - Show memory statistics")
        print("  /memory search <query> - Search memories")
        print("  /memory export         - Export memories to text file")
        print("  /memory clear          - Clear all memories (requires confirmation)")

if __name__ == "__main__":
    asyncio.run(run_agent())
