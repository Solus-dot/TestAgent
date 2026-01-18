import asyncio
import os
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Set environment variable for memory path (so tools_server.py uses same file)
MEMORY_PATH = 'memories/agent_memory.pkl'
os.environ['MEMORY_PATH'] = MEMORY_PATH

# Import after setting env var
from memory import VectorMemory

# Connect to the Llama Server
llm_client = OpenAI(base_url="http://127.0.0.1:6767/v1", api_key="TestAgent")

# Define connection to the MCP Server
script_dir = os.path.dirname(os.path.abspath(__file__))
server_params = StdioServerParameters(
    command="python3", 
    args=[os.path.join(script_dir, "tools_server.py")],
    env={**os.environ, 'MEMORY_PATH': MEMORY_PATH}
)

# Initialize memory system
memory = VectorMemory(storage_path=MEMORY_PATH)

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
            
            # Enhanced system prompt with parallel tool calling emphasis
            system_prompt = """You are an intelligent assistant with long-term memory.

CORE OPERATING RULES:
1. PARALLEL TOOL USE: You must call MULTIPLE tools in a single turn whenever possible. Do not wait for a tool result to call the next unrelated tool.
   - Bad: [call tool A] -> wait -> [call tool B]
   - Good: [call tool A, call tool B] -> process results
2. MEMORY FIRST: Always search memory [recall_memory()] before answering personal questions.
3. PROACTIVE STORAGE: If the user states a fact, preference, or goal, save it immediately [store_memory()].

MEMORY GUIDELINES:
- Types: "identity" (who they are), "preference" (likes/dislikes), "fact" (info), "goal" (plans).
- Strategy: When asked "Who am I?", search for "name", "job", "location" simultaneously.

EXAMPLES:
- User: "I love Python but hate Java."
  -> Tool Calls: [store_memory("Loves Python", "preference"), store_memory("Hates Java", "preference")]
- User: "Forget where I live."
  -> Tool Calls: [recall_memory("home address"), forget_memory(found_id)]
- User: "What's my name and the weather in Tokyo?"
  -> Tool Calls: [recall_memory("my name"), get_weather("Tokyo")]

IMPORTANT: Always call all needed tools in ONE response, not across multiple responses."""
            
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ["quit", "exit"]: 
                    break
                
                # Special memory commands
                if user_input.lower().startswith("/memory"):
                    handle_memory_command(user_input)
                    continue
                
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(history)
                messages.append({"role": "user", "content": user_input})

                # Ask LLM
                response = llm_client.chat.completions.create(
                    model="TestAgent-model",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto"
                )

                msg = response.choices[0].message
                history.append({"role": "user", "content": user_input})
                
                # Handle tool calls
                if msg.tool_calls:
                    # Show how many tools are being called
                    num_calls = len(msg.tool_calls)
                    if num_calls > 1:
                        print(f"[AGENT]  {num_calls} parallel tool calls:")
                    else:
                        print(f"[AGENT]  Tool call:")
                    
                    # Add assistant's tool call message to history
                    history.append({
                        "role": "assistant",
                        "content": msg.content or "",
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
                    })
                    
                    # Execute ALL tool calls and collect results
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

                        # Add tool result to history
                        history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output
                        })
                    
                    # Generate final response with all tool results
                    if num_calls > 1:
                        print(f"[AGENT]  Synthesizing response from {num_calls} results...")
                    
                    final = llm_client.chat.completions.create(
                        model="TestAgent-model",
                        messages=history  # Contains all tool results now
                    )
                    
                    assistant_response = final.choices[0].message.content
                    print(f"Agent: {assistant_response}")
                    history.append({"role": "assistant", "content": assistant_response})
                    
                else:
                    # No tool calls, direct response
                    assistant_response = msg.content
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
        confirm = input("[AGENT]  Are you sure you want to clear ALL memories? (yes/no): ")
        if confirm.lower() == "yes":
            memory.clear()
            print("[AGENT] All memories cleared")
        else:
            print("[AGENT] Cancelled")
    
    else:
        print("\n📚 Memory commands:")
        print("  /memory stats          - Show memory statistics")
        print("  /memory search <query> - Search memories")
        print("  /memory export         - Export memories to text file")
        print("  /memory clear          - Clear all memories (requires confirmation)")

if __name__ == "__main__":
    asyncio.run(run_agent())    