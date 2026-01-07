#!/usr/bin/env python

import asyncio
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to the Llama Server
# Make sure your llmServer.sh is running on port 6767
llm_client = OpenAI(base_url="http://127.0.0.1:6767/v1", api_key="TestAgent")

# Define connection to the Hands (MCP Server)
server_params = StdioServerParameters(
    command="python3", 
    args=["tools_server.py"],
)

async def run_agent():
    print("--- MCP Agent Connecting... ---")
    
    # Connect to the MCP Server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # Step A: Initialize and ask the server "What tools do you have?"
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
            
            # Step B: The Chat Loop
            history = [{"role": "system", "content": "You are a helpful assistant."}]
            
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ["quit", "exit"]: break
                
                history.append({"role": "user", "content": user_input})

                # 1. Ask LLM
                response = llm_client.chat.completions.create(
                    model="local-model",
                    messages=history,
                    tools=openai_tools,
                    tool_choice="auto"
                )

                msg = response.choices[0].message
                history.append(msg)

                # 2. Check for Tool Calls
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        fname = tool_call.function.name
                        fargs = tool_call.function.arguments # JSON string
                        
                        print(f"⚙️  MCP Call: {fname}({fargs})")

                        # 3. Execute via MCP Protocol
                        # We parse the arguments into a dict
                        import json
                        args_dict = json.loads(fargs)
                        
                        result = await session.call_tool(fname, arguments=args_dict)
                        
                        # Feed result back to LLM
                        # MCP returns a list of content (text/images). We grab the text.
                        tool_output = result.content[0].text
                        
                        history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output
                        })

                    # 5. Final LLM Response
                    final = llm_client.chat.completions.create(
                        model="local-model",
                        messages=history
                    )
                    print(f"Agent: {final.choices[0].message.content}")
                else:
                    print(f"Agent: {msg.content}")

if __name__ == "__main__":
    asyncio.run(run_agent())