# Detect model capabilities
from utils.ANSI import *

def get_model_info(llm_client):
    """Get information about the current model to detect capabilities"""
    try:
        model_info = llm_client.show('TestAgent-model')
        
        # 1. Check modelfile for FROM line
        modelfile = model_info.get('modelfile', '')
        base_model = None
        
        for line in modelfile.split('\n'):
            if line.strip().startswith('FROM '):
                from_value = line.strip()[5:].strip()
                # If it's not a blob path, use it
                if not from_value.startswith('/') and 'blob' not in from_value:
                    base_model = from_value
                    break
        
        # 2. Check model_info dict for other identifiers
        if not base_model:
            details = model_info.get('details', {})
            family = details.get('family', '') or details.get('families', [''])[0] if isinstance(details.get('families'), list) else ''
            if family:
                base_model = family
            
            param_size = details.get('parameter_size', '')
            if param_size:
                base_model = f"{family} ({param_size})" if family else param_size
        
        # 3. Check template for model-specific markers
        template = model_info.get('template', '')
        if template and not base_model:
            if 'qwen' in template.lower():
                base_model = 'qwen'
            elif 'gpt' in template.lower() or 'think' in template.lower():
                base_model = 'gpt-oss'
            elif 'deepseek' in template.lower():
                base_model = 'deepseek'
            elif 'llama' in template.lower():
                base_model = 'llama'
        
        if base_model:
            print(f"{YELLOW}[AGENT]{RESET} Detected base model: {base_model}")
        else:
            print(f"{YELLOW}[AGENT]{RESET} Could not determine base model, defaulting to no extended thinking")
        
        return base_model
        
    except Exception as e:
        print(f"{YELLOW}[AGENT WARNING]{RESET} Could not detect model: {e}")
        return None

# Determine if model supports extended thinking
def supports_extended_thinking(model_info):
    """Check if the model supports the 'think' parameter"""
    if not model_info:
        return False
    
    # Models/families known to support extended thinking
    thinking_indicators = ['gptoss']
    model_lower = str(model_info).lower()
    return any(indicator in model_lower for indicator in thinking_indicators)