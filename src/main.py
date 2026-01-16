import asyncio
import os
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Set environment variable for memory path (so tools_server.py uses same file)
MEMORY_PATH = "agent_memory.pkl"
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
    env={**os.environ, 'MEMORY_PATH': MEMORY_PATH}  # Pass env var to subprocess
)

# Initialize memory system (same file as tools_server)
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
            print(f"Memory tools: store_memory, recall_memory, list_recent_memories, forget_memory")
            
            # The Chat Loop
            history = []
            
            # Enhanced system prompt
            system_prompt = """You are a helpful assistant with access to tools, including a long-term memory system.

MEMORY USAGE GUIDELINES:
- When users share personal information (name, preferences, facts), use store_memory() to remember it
- Before answering questions about the user, use recall_memory() to check what you know
- Be proactive: if something seems worth remembering, store it without asking
- When recalling, search with specific queries (e.g., "user's name" not just "name")

MEMORY TYPES:
- identity: Name, age, job, location, personal details
- preference: Likes, dislikes, favorites, opinions
- fact: General information, skills, experiences
- goal: User's objectives, projects, learning goals

Examples:
- User: "My name is Alex" → store_memory("User's name is Alex", "identity", "high")
- User: "What's my name?" → recall_memory("user's name", "identity", 1) first, then respond
- User: "I love Python" → store_memory("User loves Python programming", "preference", "normal")

Always use recall_memory() before claiming you don't know something about the user."""
            
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
                    model="local-model",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto"
                )

                msg = response.choices[0].message
                history.append({"role": "user", "content": user_input})
                
                # Handle tool calls
                if msg.tool_calls:
                    history.append({
                        "role": "assistant",
                        "content": msg.content,
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
                    
                    for tool_call in msg.tool_calls:
                        fname = tool_call.function.name
                        fargs = tool_call.function.arguments
                        
                        print(f"⚙️ MCP Call: {fname}({fargs})")

                        # Execute via MCP Protocol
                        # We parse the arguments into a dict
                        import json
                        args_dict = json.loads(fargs)
                        
                        result = await session.call_tool(fname, arguments=args_dict)
                        
                        # Feed result back to LLM
                        # MCP returns a list of content (text/images). We grab the text.
                        tool_output = result.content[0].text
                        print(f"  > Result: {tool_output[:200]}...")

                        history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output
                        })

                    # Final LLM Response
                    final = llm_client.chat.completions.create(
                        model="local-model",
                        messages=messages + [
                            {
                                "role": "assistant",
                                "content": msg.content,
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
                        ] + [
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result.content[0].text
                            } for tc in msg.tool_calls
                        ]
                    )
                    assistant_response = final.choices[0].message.content
                    print(f"Agent: {assistant_response}")
                    history.append({"role": "assistant", "content": assistant_response})
                else:
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
        print("\n📊 Memory Statistics:")
        print(f"Total memories: {stats['total_memories']}")
        if stats['total_memories'] > 0:
            print(f"Oldest: {stats['oldest']['text']} ({stats['oldest']['age_days']:.1f} days old)")
            print(f"Newest: {stats['newest']['text']} ({stats['newest']['age_days']:.1f} days old)")
            print(f"Most accessed: {stats['most_accessed']['text']} ({stats['most_accessed']['count']} times)")
            print(f"Types: {stats['types']}")
    
    elif parts[1] == "search" and len(parts) > 2:
        query = " ".join(parts[2:])
        results = memory.search(query, top_k=5)
        print(f"\n🔍 Search results for '{query}':")
        for r in results:
            print(f"  [{r['score']:.2f}] {r['text']}")
    
    elif parts[1] == "export":
        memory.export_txt()
    
    elif parts[1] == "clear":
        confirm = input("Are you sure you want to clear ALL memories? (yes/no): ")
        if confirm.lower() == "yes":
            memory.clear()
    
    else:
        print("\nMemory commands:")
        print("  /memory stats - Show memory statistics")
        print("  /memory search <query> - Search memories")
        print("  /memory export - Export memories to text file")
        print("  /memory clear - Clear all memories")

if __name__ == "__main__":
    asyncio.run(run_agent())