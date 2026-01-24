import asyncio
import os
import re
import sys
import json
from ollama import Client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set environment variable for memory path 
MEMORY_PATH = 'memories/agent_memory.pkl'
os.environ['MEMORY_PATH'] = MEMORY_PATH

# Import after setting env var
from core.memory import VectorMemory
from core.prompts import SYSTEM_PROMPT

# --- Configuration ---
DEBUG_MODE = False  # Set to True to see full inputs/outputs

llm_client = Client(host='http://127.0.0.1:6767')

script_dir = os.path.dirname(os.path.abspath(__file__))
server_params = StdioServerParameters(
    command="python3", 
    args=[os.path.join(os.path.dirname(script_dir), "tools", "tools_server.py")],
    env={**os.environ, 'MEMORY_PATH': MEMORY_PATH}
)

# Initialize memory system
memory = VectorMemory(storage_path=MEMORY_PATH)

# --- ANSI Color Codes ---
GREY = "\033[90m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def debug_print(label, data):
    """Helper to print raw data if DEBUG_MODE is on"""
    if DEBUG_MODE:
        print(f"\n{YELLOW}--- DEBUG: {label} ---{RESET}")
        try:
            if hasattr(data, 'model_dump'):
                print(json.dumps(data.model_dump(), indent=2, default=str))
            elif hasattr(data, '__dict__'):
                print(json.dumps(data.__dict__, indent=2, default=str))
            else:
                print(json.dumps(data, indent=2, default=str))
        except Exception:
            print(str(data))
        print(f"{YELLOW}-----------------------{RESET}\n")

def extract_thought(text):
    if not text: return None
    think_pattern = r'<think>(.*?)</think>'
    match = re.search(think_pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_final_response(text):
    if not text: return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    final_pattern = r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)'
    match = re.search(final_pattern, text, re.DOTALL)
    if match: return match.group(1).strip()
    
    if '<|' not in text: return text.strip()
    cleaned = re.sub(r'<\|.*?\|>', '', text)
    return cleaned.strip()

def print_ollama_metrics(response):
    total_dur = getattr(response, 'total_duration', 0)
    load_dur = getattr(response, 'load_duration', 0)
    eval_count = getattr(response, 'eval_count', 0)
    eval_dur = getattr(response, 'eval_duration', 1)
    
    if total_dur == 0: return

    tks = eval_count / (eval_dur / 1e9)
    
    print(f"{CYAN}\nMetrics: {total_dur/1e9:.2f}s total (Load: {load_dur/1e9:.2f}s)")
    print(f"Speed:   {eval_count} tokens @ {tks:.2f} t/s{CYAN}")

async def run_agent():
    print("--- MCP Agent Connecting (Native Ollama) ---")
    if DEBUG_MODE:
        print(f"{YELLOW}DEBUG MODE IS ON{RESET}")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            await session.initialize()
            mcp_tools_list = await session.list_tools()
            
            agent_tools = []
            for tool in mcp_tools_list.tools:
                agent_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            
            print(f"Loaded {len(agent_tools)} tools.")
            
            # Print Tools Schema for Debugging
            debug_print("TOOLS SCHEMA", agent_tools)

            history = []
            
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ["quit", "exit"]: break
                
                if user_input.lower().startswith("/memory"):
                    handle_memory_command(user_input)
                    continue
                
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages.extend(history)
                messages.append({"role": "user", "content": user_input})

                debug_print("INPUT MESSAGES", messages)

                # Call Ollama Native API
                response = llm_client.chat(
                    model="TestAgent-model",
                    messages=messages,
                    think="high",
                    tools=agent_tools,
                )

                debug_print("RAW RESPONSE", response)

                msg = response.message
                content = msg.content or ""
                tool_calls = msg.tool_calls or []
                
                thinking_content = getattr(msg, 'thinking', None)
                if not thinking_content:
                    thinking_content = extract_thought(content)

                history.append({"role": "user", "content": user_input})
                
                if tool_calls:
                    print(f"[AGENT] {len(tool_calls)} tool calls...")
                    
                    assistant_content = extract_final_response(content)
                    
                    history_tool_calls = []
                    for tc in tool_calls:
                        tc_dict = tc.model_dump() if hasattr(tc, 'model_dump') else tc
                        history_tool_calls.append({
                            "type": "function",
                            "function": {
                                "name": tc_dict['function']['name'],
                                "arguments": tc_dict['function']['arguments']
                            }
                        })

                    history.append({
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": history_tool_calls
                    })
                    
                    for i, tool_call in enumerate(tool_calls, 1):
                        if hasattr(tool_call, 'function'):
                            fname = tool_call.function.name
                            fargs = tool_call.function.arguments
                        else:
                            fname = tool_call['function']['name']
                            fargs = tool_call['function']['arguments']
                        
                        print(f"[AGENT] Executing {fname}({fargs})")
                        
                        result = await session.call_tool(fname, arguments=fargs)
                        tool_output = result.content[0].text
                        
                        preview = tool_output[:100] + "..." if len(tool_output) > 100 else tool_output
                        print(f"[AGENT] Result: {preview}")

                        history.append({
                            "role": "tool",
                            "content": tool_output,
                            "name": fname 
                        })
                    
                    print(f"[AGENT] Synthesizing final response...")
                    debug_print("SYNTHESIS INPUT", history)

                    final_response = llm_client.chat(
                        model="TestAgent-model",
                        messages=history
                    )
                    debug_print("SYNTHESIS RESPONSE", final_response)
                    
                    final_msg = final_response.message
                    raw_response = final_msg.content
                    
                    final_thinking = getattr(final_msg, 'thinking', None)
                    if not final_thinking:
                        final_thinking = extract_thought(raw_response)
                        
                    if final_thinking:
                        print(f"\n{GREY}[REASONING]\n{final_thinking}\n[END REASONING]{RESET}\n")

                    assistant_response = extract_final_response(raw_response)
                    print(f"Agent: {assistant_response}")
                    
                    print_ollama_metrics(final_response)
                    history.append({"role": "assistant", "content": assistant_response})
                    
                else:
                    if thinking_content:
                        print(f"\n{GREY}[REASONING]\n{thinking_content}\n[END REASONING]{RESET}\n")

                    assistant_response = extract_final_response(content)
                    print(f"Agent: {assistant_response}")
                    
                    print_ollama_metrics(response)
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
