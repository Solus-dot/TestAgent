#!/usr/bin/env python

import asyncio
import time
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to the Llama Server
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
            history = [{"role": "system", "content": """You are a helpful assistant with access to tools.
                        When you call a tool and receive results, use those results to answer the user's question.
                        Never say you can't access something if a tool has already provided the information."""}]
            
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ["quit", "exit"]: 
                    break
                
                history.append({"role": "user", "content": user_input})

                # Track timing for initial LLM call
                start_time = time.time()
                first_token_time = None
                total_tokens = 0

                # Ask LLM
                response = llm_client.chat.completions.create(
                    model="TestAgent-model",
                    messages=history,
                    tools=openai_tools,
                    tool_choice="auto",
                    stream=True  # Enable streaming to track first token
                )

                # Collect the streamed response
                collected_messages = []
                tool_calls_data = []
                current_tool_call = None
                
                for chunk in response:
                    # Track first token timing
                    if first_token_time is None and chunk.choices:
                        first_token_time = time.time()
                    
                    delta = chunk.choices[0].delta if chunk.choices else None
                    
                    if delta:
                        # Collect content
                        if delta.content:
                            collected_messages.append(delta.content)
                        
                        # Collect tool calls
                        if delta.tool_calls:
                            for tc_delta in delta.tool_calls:
                                if tc_delta.index is not None:
                                    # Ensure we have a slot for this tool call
                                    while len(tool_calls_data) <= tc_delta.index:
                                        tool_calls_data.append({
                                            "id": None,
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""}
                                        })
                                    
                                    current_tool_call = tool_calls_data[tc_delta.index]
                                    
                                    if tc_delta.id:
                                        current_tool_call["id"] = tc_delta.id
                                    if tc_delta.function:
                                        if tc_delta.function.name:
                                            current_tool_call["function"]["name"] = tc_delta.function.name
                                        if tc_delta.function.arguments:
                                            current_tool_call["function"]["arguments"] += tc_delta.function.arguments

                end_time = time.time()
                
                # Calculate metrics
                total_time = end_time - start_time
                time_to_first_token = (first_token_time - start_time) if first_token_time else 0
                
                # Reconstruct the message
                full_content = "".join(collected_messages)
                
                # Create message object
                class MockMessage:
                    def __init__(self, content, tool_calls):
                        self.content = content
                        self.tool_calls = []
                        if tool_calls:
                            for tc in tool_calls:
                                class MockToolCall:
                                    def __init__(self, id, function_name, function_args):
                                        self.id = id
                                        self.type = "function"
                                        self.function = type('obj', (object,), {
                                            'name': function_name,
                                            'arguments': function_args
                                        })()
                                
                                self.tool_calls.append(
                                    MockToolCall(tc["id"], tc["function"]["name"], tc["function"]["arguments"])
                                )
                
                msg = MockMessage(full_content, tool_calls_data if tool_calls_data else None)
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
                    ] if msg.tool_calls else None
                })

                # Show metrics for initial response
                tokens_generated = len(full_content.split()) if full_content else len(str(tool_calls_data).split())
                tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0
                
                print(f"\n⏱️  Time to first token: {time_to_first_token:.2f}s | Total time: {total_time:.2f}s | Speed: ~{tokens_per_sec:.1f} tokens/s")

                # Check for Tool Calls
                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        fname = tool_call.function.name
                        fargs = tool_call.function.arguments
                        
                        print(f"⚙️  MCP Call: {fname}({fargs})")

                        # Execute via MCP Protocol
                        import json
                        args_dict = json.loads(fargs)
                        
                        result = await session.call_tool(fname, arguments=args_dict)
                        
                        # Feed result back to LLM
                        tool_output = result.content[0].text
                        print(f"  > [Tool Output]: {tool_output[:200]}{'...' if len(tool_output) > 200 else ''}")

                        history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output
                        })

                    # Final LLM Response (also with streaming)
                    final_start = time.time()
                    final_first_token = None
                    
                    final_response = llm_client.chat.completions.create(
                        model="local-model",
                        messages=history,
                        stream=True
                    )
                    
                    final_content = []
                    for chunk in final_response:
                        if final_first_token is None and chunk.choices:
                            final_first_token = time.time()
                        
                        delta = chunk.choices[0].delta if chunk.choices else None
                        if delta and delta.content:
                            final_content.append(delta.content)
                    
                    final_end = time.time()
                    final_text = "".join(final_content)
                    
                    # Calculate final metrics
                    final_total_time = final_end - final_start
                    final_time_to_first = (final_first_token - final_start) if final_first_token else 0
                    final_tokens = len(final_text.split())
                    final_tokens_per_sec = final_tokens / final_total_time if final_total_time > 0 else 0
                    
                    print(f"Agent: {final_text}")
                    print(f"⏱️  Time to first token: {final_time_to_first:.2f}s | Total time: {final_total_time:.2f}s | Speed: ~{final_tokens_per_sec:.1f} tokens/s")
                    
                    history.append({"role": "assistant", "content": final_text})
                else:
                    if msg.content:
                        print(f"Agent: {msg.content}")

if __name__ == "__main__":
    asyncio.run(run_agent())