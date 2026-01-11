import asyncio
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to the LMStudio Server
llm_client = OpenAI(base_url="http://127.0.0.1:6767/v1", api_key="TestAgent")

# Define connection to the MCP Server
server_params = StdioServerParameters(
    command="python3", 
    args=["src/tools_server.py"],
)

async def run_agent():
    print("--- MCP Agent Connecting... ---")
    
    # Connect to the MCP Server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # Initialize and check server for tool list
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
            history = [{"role": "system", "content": """You are a helpful assistant with access to tools. When you call a tool and receive results, use those results to answer the user's question. Never say you can't access something if a tool has already provided the information."""}]
            
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ["quit", "exit"]: break
                
                history.append({"role": "user", "content": user_input})

                # Ask LLM
                response = llm_client.chat.completions.create(
                    model="local-model",
                    messages=history,
                    tools=openai_tools,
                    tool_choice="auto"
                )

                msg = response.choices[0].message
                history.append(msg)

                # Check for Tool Calls
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        fname = tool_call.function.name
                        fargs = tool_call.function.arguments # JSON string
                        
                        print(f"⚙️  MCP Call: {fname}({fargs})")

                        # Execute via MCP Protocol
                        # We parse the arguments into a dict
                        import json
                        args_dict = json.loads(fargs)
                        
                        result = await session.call_tool(fname, arguments=args_dict)
                        
                        # Feed result back to LLM
                        # MCP returns a list of content (text/images). We grab the text.
                        tool_output = result.content[0].text
                        print(f"  > [Tool Output]: {tool_output}")

                        
                        history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output
                        })

                    # Final LLM Response
                    final = llm_client.chat.completions.create(
                        model="local-model",
                        messages=history
                    )
                    print(f"Agent: {final.choices[0].message.content}")
                else:
                    print(f"Agent: {msg.content}")

if __name__ == "__main__":
    asyncio.run(run_agent())