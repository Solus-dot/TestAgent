#!/usr/bin/env python3
import sys
import json
import re

def clean_text(text):
    """Replace literal \n with actual newlines"""
    return text.replace('\\n', '\n').strip()

def parse_prompt_sections(text):
    """Parse the prompt into structured sections"""
    sections = {
        'system': [],
        'developer': [],
        'user': [],
        'assistant': [],
        'tools': []
    }
    
    # Find all sections
    system_matches = re.findall(r'<\|start\|>system<\|message\|>(.*?)<\|end\|>', text, re.DOTALL)
    dev_matches = re.findall(r'<\|start\|>developer<\|message\|>(.*?)<\|end\|>', text, re.DOTALL)
    user_matches = re.findall(r'<\|start\|>user<\|message\|>(.*?)<\|end\|>', text, re.DOTALL)
    assistant_matches = re.findall(r'<\|start\|>assistant<\|channel\|>.*?<\|message\|>(.*?)<\|end\|>', text, re.DOTALL)
    
    if system_matches:
        sections['system'] = [clean_text(m) for m in system_matches]
    if dev_matches:
        sections['developer'] = [clean_text(m) for m in dev_matches]
    if user_matches:
        sections['user'] = [clean_text(m) for m in user_matches]
    if assistant_matches:
        sections['assistant'] = [clean_text(m) for m in assistant_matches]
    
    # Extract tool definitions
    tools_match = re.search(r'namespace functions \{(.*?)\} // namespace functions', text, re.DOTALL)
    if tools_match:
        sections['tools'] = clean_text(tools_match.group(1))
    
    return sections

def parse_output_sections(text):
    """Parse the output into channels and content"""
    sections = {
        'analysis': None,
        'commentary': None,
        'final': None,
        'tool_call': None
    }
    
    # Extract analysis channel
    analysis_match = re.search(r'<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)', text, re.DOTALL)
    if analysis_match:
        sections['analysis'] = clean_text(analysis_match.group(1))
    
    # Extract commentary (tool calls)
    commentary_match = re.search(r'<\|channel\|>commentary to=functions\.(\w+).*?<\|message\|>(.*?)(?:<\|end\|>|$)', text, re.DOTALL)
    if commentary_match:
        tool_name = commentary_match.group(1)
        tool_args = commentary_match.group(2).replace('<|constrain|>json', '').strip()
        sections['tool_call'] = {'name': tool_name, 'args': tool_args}
    
    # Extract final channel
    final_match = re.search(r'<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|$)', text, re.DOTALL)
    if final_match:
        sections['final'] = clean_text(final_match.group(1))
    
    return sections

for line in sys.stdin:
    try:
        data = json.loads(line.strip())
        log_type = data.get("data", {}).get("type", "unknown")
        
        if log_type == "llm.prediction.input":
            print("\nPROMPT INPUT")
            print("="*10)
            
            input_text = data["data"].get("input", "")
            sections = parse_prompt_sections(input_text)
            
            # System prompt
            if sections['system']:
                print("\nSYSTEM PROMPT:")
                print("-" * 10)
                for msg in sections['system']:
                    print(msg)
            
            # Developer instructions
            if sections['developer']:
                print("\nDEVELOPER INSTRUCTIONS:")
                print("-" * 10)
                for msg in sections['developer']:
                    # Show just the first 500 chars if it's really long
                    if len(msg) > 500:
                        print(msg[:500] + "\n... (truncated)")
                    else:
                        print(msg)
            
            # Available tools
            if sections['tools']:
                print("\nAVAILABLE TOOLS:")
                print("-" * 10)
                print(sections['tools'])
            
            # Conversation history (exclude the current/last message)
            if sections['user'] and len(sections['user']) > 1:
                print("\nCONVERSATION HISTORY:")
                print("-" * 10)
                
                # Exclude the current message from users
                historical_users = sections['user'][:-1]
                historical_assistants = sections['assistant']

                for i in range(len(historical_users)):
                    print(f"\nUser: {historical_users[i]}")
                    print(f"Assistant: {historical_assistants[i]}")

            # Current user message (always show the latest)
            if sections['user']:
                print("\nCURRENT USER MESSAGE:")
                print("="*10)
                print(f"{sections['user'][-1]}")
        
        elif log_type == "llm.prediction.output":
            stats = data["data"].get("stats", {})
            output_text = data["data"].get("output", "")
            print("\nMODEL OUTPUT")
            print("="*10)
            
            sections = parse_output_sections(output_text)
            
            # Model thinking (analysis)
            if sections['analysis']:
                print("\nMODEL THINKING (Analysis Channel):")
                print("-" * 10)
                print(sections['analysis'])
            
            # Tool call
            if sections['tool_call']:
                print("\nTOOL CALL (Commentary Channel):")
                print("-" * 10)
                tool = sections['tool_call']
                print(f"Function: {tool['name']}")
                print(f"Arguments: {tool['args']}")
            
            # Final response
            if sections['final']:
                print("\nFINAL RESPONSE:")
                print("-" * 10)
                print(sections['final'])
            
            # Show raw output if nothing was parsed
            if not any([sections['analysis'], sections['tool_call'], sections['final']]):
                print("\nRAW OUTPUT:")
                print("-" * 10)
                print(clean_text(output_text))
            
            # Performance stats
            print("\nPERFORMANCE STATS")
            print("="*10)
            print(f"Speed: {stats.get('tokensPerSecond', 0):.1f} tok/s")
            print(f"Time to first token: {stats.get('timeToFirstTokenSec', 0):.3f}s")
            print(f"Total generation time: {stats.get('totalTimeSec', 0):.3f}s")
            print(f"Prompt tokens: {stats.get('promptTokensCount', 0)}")
            print(f"Generated tokens: {stats.get('predictedTokensCount', 0)}")
            print(f"Total tokens: {stats.get('totalTokensCount', 0)}")
            print(f"Stop reason: {stats.get('stopReason', 'unknown')}")
            print("="*10 + "\n")
    
    except json.JSONDecodeError:
        continue
    except Exception as e:
        print(f"Error parsing log: {e}", file=sys.stderr)
        # Print raw line for debugging
        print(f"Raw line: {line[:200]}...", file=sys.stderr)
