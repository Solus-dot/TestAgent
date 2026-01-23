import os
import sys
import json
import glob
import time
import argparse
import uvicorn
import io
import re
from typing import List, Optional, Dict, Union
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager, redirect_stderr

# --- Colors ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    GREY = '\033[90m'

# --- Configuration ---
HOST = "127.0.0.1"
PORT = 6767
MODEL_ENGINE = None
VERBOSE_MODE = False

# --- Pydantic Models ---
class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[Dict]] = None

class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Union[str, Dict]] = None

# --- TOOL FIXER MIDDLEWARE ---
class ToolFixer:
    """
    Mimics LM Studio's logic: 
    If response is raw JSON but not a formal tool call, 
    try to match it to a requested tool schema.
    """
    @staticmethod
    def fix(raw_content: str, available_tools: List[Dict]) -> Optional[List[Dict]]:
        if not available_tools or not raw_content:
            return None

        clean_content = raw_content.strip()
        
        # Remove markdown code blocks
        if clean_content.startswith("```"):
            lines = clean_content.split("\n")
            clean_content = "\n".join(lines[1:-1]) if len(lines) > 2 else clean_content
            if clean_content.startswith("json"):
                clean_content = clean_content[4:].strip()
        
        # Remove any reasoning before JSON
        if "{" in clean_content:
            clean_content = clean_content[clean_content.find("{"):]
        if "}" in clean_content:
            clean_content = clean_content[:clean_content.rfind("}")+1]
        
        try:
            data = json.loads(clean_content)
        except json.JSONDecodeError:
            return None

        # If not a dict, can't be tool arguments
        if not isinstance(data, dict):
            return None

        # 1. Check if model explicitly outputted "tool": "name"
        if "tool" in data and "arguments" in data:
            return [{
                "id": f"call_{int(time.time()*1000)}",
                "type": "function",
                "function": {"name": data["tool"], "arguments": json.dumps(data["arguments"])}
            }]

        # 2. Heuristic: Match JSON keys to Tool Parameters
        best_match = None
        max_matches = 0
        best_coverage = 0

        for tool in available_tools:
            tool_name = tool['function']['name']
            params = tool['function'].get('parameters', {}).get('properties', {})
            required = tool['function'].get('parameters', {}).get('required', [])
            
            # Count matching keys
            matching_keys = [key for key in data.keys() if key in params]
            matches = len(matching_keys)
            
            # Check if required params are present
            has_required = all(req in data for req in required)
            
            # Calculate coverage (what % of data keys match this tool)
            coverage = matches / len(data.keys()) if data.keys() else 0
            
            # Prefer tools where:
            # 1. All required params are present
            # 2. High coverage of data keys
            # 3. More matches overall
            if has_required and coverage > 0.5:
                # Better match if: higher coverage, or same coverage but more matches
                if coverage > best_coverage or (coverage == best_coverage and matches > max_matches):
                    max_matches = matches
                    best_coverage = coverage
                    best_match = tool_name

        if best_match:
            if VERBOSE_MODE:
                print(f"{Colors.WARNING}>>> [MIDDLEWARE] Auto-converted JSON to ToolCall: {best_match} (coverage: {best_coverage:.0%}){Colors.ENDC}")
            return [{
                "id": f"call_{int(time.time()*1000)}",
                "type": "function",
                "function": {"name": best_match, "arguments": json.dumps(data)}
            }]
        
        return None

# --- Engine Interfaces ---
class EngineInterface:
    def chat(self, request: ChatCompletionRequest) -> Dict:
        raise NotImplementedError

class GGUFEngine(EngineInterface):
    def __init__(self, model_path, n_ctx, verbose):
        self.verbose = verbose
        try:
            from llama_cpp import Llama
            print(f"{Colors.BLUE}Loading GGUF model: {model_path}...{Colors.ENDC}")
            self.llm = Llama(
                model_path=model_path, 
                n_ctx=n_ctx, 
                verbose=self.verbose,  # Use verbose flag
                n_gpu_layers=-1
            )
            print(f"{Colors.GREEN}GGUF Model Loaded Successfully.{Colors.ENDC}")
        except ImportError:
            sys.exit("llama-cpp-python not installed. Run: pip install llama-cpp-python")

    def chat(self, request: ChatCompletionRequest) -> Dict:
        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        tools = request.tools if request.tools else None
        
        # History Logging
        if self.verbose:
            print(f"\n{Colors.GREY}--- [GGUF] HISTORY ({len(messages)} msgs) ---{Colors.ENDC}")
            for i, m in enumerate(messages, 1):
                role_color = Colors.CYAN if m['role'] == 'user' else Colors.GREEN if m['role'] == 'assistant' else Colors.GREY
                content_preview = str(m.get('content', ''))[:100].replace('\n', ' ')
                print(f"{role_color}[{i}] {m['role'].upper()}: {content_preview}...{Colors.ENDC}")

        # Capture llama.cpp verbose output
        stderr_capture = io.StringIO()
        with redirect_stderr(stderr_capture):
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=tools,
                tool_choice=request.tool_choice,
                stream=False
            )
        
        llama_output = stderr_capture.getvalue()
        
        # --- MIDDLEWARE INTERCEPTION ---
        choice = response['choices'][0]
        msg = choice['message']
        
        # Calculate metrics
        usage = response.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        # If no tool calls detected by library, check for "Raw JSON"
        if not msg.get('tool_calls') and tools:
            raw_content = msg.get('content', '')
            fixed_tools = ToolFixer.fix(raw_content, tools)
            if fixed_tools:
                if self.verbose:
                    print(f"{Colors.WARNING}>>> [MIDDLEWARE] Fixed tool call: {fixed_tools[0]['function']['name']}{Colors.ENDC}")
                
                # Update the message properly
                msg['tool_calls'] = fixed_tools
                msg['content'] = ""  # Empty string, not None
                choice['finish_reason'] = 'tool_calls'
                
                # Update the choice in response
                response['choices'][0] = choice

        # Logging
        if self.verbose:
            print(f"\n{Colors.GREY}--- [GGUF] RESPONSE ---{Colors.ENDC}")
            
            # Extract performance metrics from llama.cpp output
            tokens_per_sec = None
            total_time = None
            
            # Parse llama.cpp verbose output for performance metrics
            for line in llama_output.split('\n'):
                # Look for tokens per second pattern
                tps_match = re.search(r'(\d+\.?\d*)\s+tokens?\s+per\s+second', line, re.IGNORECASE)
                if tps_match:
                    tokens_per_sec = float(tps_match.group(1))
                
                # Look for timing patterns
                time_match = re.search(r'total time[:\s]+(\d+\.?\d*)\s*(?:ms|s)', line, re.IGNORECASE)
                if time_match:
                    total_time = float(time_match.group(1))
                    if 'ms' in line.lower():
                        total_time = total_time / 1000.0  # Convert to seconds
            
            # Display performance metrics
            if tokens_per_sec or total_time or completion_tokens > 0:
                print(f"{Colors.BLUE}Performance:{Colors.ENDC}")
                if total_time:
                    print(f"  Total time: {total_time:.2f}s")
                if tokens_per_sec:
                    print(f"  Speed: {tokens_per_sec:.1f} tokens/s")
                print(f"  Tokens: {completion_tokens} generated | {prompt_tokens} prompt | {total_tokens} total")
            
            # Response content
            if msg.get('tool_calls'):
                print(f"{Colors.WARNING}Tool Calls:{Colors.ENDC}")
                for tc in msg['tool_calls']:
                    print(f"  {tc['function']['name']}({tc['function']['arguments']})")
            else:
                content = msg.get('content', '')
                
                # Handle reasoning
                if "<think>" in content and "</think>" in content:
                    reasoning = content.split("<think>")[1].split("</think>")[0].strip()
                    actual_response = content.split("</think>")[1].strip() if "</think>" in content else ""
                    
                    print(f"{Colors.BLUE}Reasoning:{Colors.ENDC}")
                    print(f"{Colors.GREY}{reasoning[:200]}...{Colors.ENDC}" if len(reasoning) > 200 else f"{Colors.GREY}{reasoning}{Colors.ENDC}")
                    print(f"\n{Colors.CYAN}Response:{Colors.ENDC}")
                    print(f"{Colors.CYAN}{actual_response}{Colors.ENDC}")
                else:
                    print(f"{Colors.CYAN}Response:{Colors.ENDC}")
                    preview = content[:300] + "..." if len(content) > 300 else content
                    print(f"{Colors.CYAN}{preview}{Colors.ENDC}")

        return response

class MLXEngine(EngineInterface):
    def __init__(self, model_path, verbose):
        self.verbose = verbose
        try:
            from mlx_lm import load, generate
            print(f"{Colors.BLUE}Loading MLX model: {model_path}...{Colors.ENDC}")
            self.load_fn = load
            self.generate_fn = generate
            self.model, self.tokenizer = self.load_fn(model_path)
            print(f"{Colors.GREEN}MLX Model Loaded Successfully.{Colors.ENDC}")
        except ImportError:
            sys.exit("mlx-lm not installed. Run: pip install mlx-lm")

    def chat(self, request: ChatCompletionRequest) -> Dict:
        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except:
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

        if self.verbose:
            print(f"\n{Colors.GREY}--- [MLX] PROMPT ---{Colors.ENDC}")
            preview = prompt[:300] + "..." if len(prompt) > 300 else prompt
            print(f"{Colors.GREY}{preview}{Colors.ENDC}")

        # Run inference with timing
        start_time = time.perf_counter()
        response_text = self.generate_fn(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=request.max_tokens, 
            verbose=False
        )
        end_time = time.perf_counter()
        
        total_time = end_time - start_time

        # --- MIDDLEWARE INTERCEPTION ---
        tool_calls = None
        finish_reason = "stop"
        final_content = response_text

        if request.tools:
            fixed_tools = ToolFixer.fix(response_text, request.tools)
            if fixed_tools:
                tool_calls = fixed_tools
                final_content = ""
                finish_reason = "tool_calls"

        # Calculate metrics
        generated_tokens = len(self.tokenizer.encode(response_text))
        tokens_per_sec = generated_tokens / total_time if total_time > 0 else 0

        # Logging
        if self.verbose:
            print(f"\n{Colors.GREY}--- [MLX] RESPONSE ---{Colors.ENDC}")
            
            # Performance metrics
            print(f"{Colors.BLUE}Performance:{Colors.ENDC}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Speed: {tokens_per_sec:.1f} tokens/s")
            print(f"  Tokens: {generated_tokens} generated")
            
            # Response content
            if tool_calls:
                print(f"{Colors.WARNING}Tool Calls:{Colors.ENDC}")
                for tc in tool_calls:
                    print(f"  {tc['function']['name']}({tc['function']['arguments']})")
            else:
                print(f"{Colors.CYAN}Response:{Colors.ENDC}")
                preview = response_text[:300] + "..." if len(response_text) > 300 else response_text
                print(f"{Colors.CYAN}{preview}{Colors.ENDC}")

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mlx-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", 
                    "content": final_content,
                    "tool_calls": tool_calls
                },
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": -1, 
                "completion_tokens": generated_tokens, 
                "total_tokens": -1
            }
        }

# --- Model Management ---
def scan_models(models_dir: str):
    """Scan for GGUF and MLX models in the given directory"""
    full_path = os.path.expanduser(models_dir)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        return []
    
    models = []
    
    # Scan for GGUF models
    for p in glob.glob(os.path.join(full_path, "**", "*.gguf"), recursive=True):
        models.append({
            "type": "GGUF", 
            "path": p, 
            "name": os.path.relpath(p, full_path)
        })
    
    # Scan for MLX models (directories with config.json)
    for p in glob.glob(os.path.join(full_path, "**", "config.json"), recursive=True):
        models.append({
            "type": "MLX", 
            "path": os.path.dirname(p), 
            "name": os.path.relpath(os.path.dirname(p), full_path)
        })
    
    return models

def select_model_menu(models):
    """Interactive model selection menu"""
    print(f"\n{Colors.HEADER}--- AVAILABLE MODELS ---{Colors.ENDC}")
    for i, m in enumerate(models):
        print(f"{Colors.BOLD}{i+1}. [{m['type']}]{Colors.ENDC} {m['name']}")
    
    while True:
        try:
            choice = int(input(f"\n{Colors.CYAN}Select model (1-{len(models)}): {Colors.ENDC}")) - 1
            if 0 <= choice < len(models):
                return models[choice]
            else:
                print(f"{Colors.FAIL}Invalid choice. Try again.{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.FAIL}Please enter a number.{Colors.ENDC}")
        except KeyboardInterrupt:
            print(f"\n{Colors.FAIL}Cancelled.{Colors.ENDC}")
            sys.exit(0)

# --- FastAPI Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown handler"""
    print(f"\n{Colors.GREEN}Server started on http://{HOST}:{PORT}{Colors.ENDC}")
    print(f"{Colors.GREY}OpenAI-compatible endpoint: http://{HOST}:{PORT}/v1/chat/completions{Colors.ENDC}")
    if VERBOSE_MODE:
        print(f"{Colors.WARNING}Verbose mode: ENABLED{Colors.ENDC}")
    print(f"{Colors.GREY}Press CTRL+C to stop{Colors.ENDC}\n")
    yield
    print(f"\n{Colors.FAIL}Server stopped.{Colors.ENDC}")

app = FastAPI(lifespan=lifespan)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint"""
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    if VERBOSE_MODE:
        print(f"\n{Colors.HEADER}{'='*50}{Colors.ENDC}")
        print(f"{Colors.HEADER}REQUEST START{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
    
    response = MODEL_ENGINE.chat(ChatCompletionRequest(**body))
    
    if VERBOSE_MODE:
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
        print(f"{Colors.HEADER}REQUEST END{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}\n")
    
    return response

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [{
            "id": "TestAgent-model",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local"
        }]
    }

# --- Main Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local LLM Server with GGUF and MLX support")
    parser.add_argument("--models-dir", default="~/models", help="Directory to scan for models")
    parser.add_argument("--ctx", type=int, default=8192, help="Context window size (GGUF only)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging with performance metrics")
    args = parser.parse_args()
    
    VERBOSE_MODE = args.verbose
    
    print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
    print(f"{Colors.HEADER}Local LLM Server{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
    
    if VERBOSE_MODE:
        print(f"{Colors.WARNING}Verbose Mode: ENABLED{Colors.ENDC}")
    
    # Scan for models
    models = scan_models(args.models_dir)
    
    if not models:
        print(f"{Colors.FAIL}No models found in {args.models_dir}{Colors.ENDC}")
        print(f"{Colors.GREY}Place .gguf files or MLX model directories there.{Colors.ENDC}")
        sys.exit(1)
    
    # Select model
    selected = select_model_menu(models)
    
    # Initialize engine
    if selected['type'] == 'GGUF':
        MODEL_ENGINE = GGUFEngine(selected['path'], args.ctx, VERBOSE_MODE)
    elif selected['type'] == 'MLX':
        MODEL_ENGINE = MLXEngine(selected['path'], VERBOSE_MODE)
    
    # Run server
    uvicorn.run(app, host=HOST, port=PORT, log_level="error")