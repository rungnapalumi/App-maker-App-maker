"""
Token counting wrapper for OpenAI client
Tracks exact usage from API responses for both real and mock clients
"""

from typing import Any, Dict, List
import time


class TokenCountingWrapper:
    """
    Wraps an OpenAI client to track exact token usage and latency.
    Works with both real OpenAI clients and mock clients.
    """
    
    def __init__(self, client):
        """
        Args:
            client: OpenAI client (real or mock)
        """
        self.client = client
        self.calls: List[Dict[str, Any]] = []
        
    def reset_tracking(self):
        """Clear tracked calls"""
        self.calls = []
    
    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all calls"""
        return sum(call["input_tokens"] for call in self.calls)
    
    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all calls"""
        return sum(call["output_tokens"] for call in self.calls)
    
    @property
    def total_latency_ms(self) -> int:
        """Total latency across all calls"""
        return sum(call["latency_ms"] for call in self.calls)
    
    @property
    def call_count(self) -> int:
        """Number of API calls made"""
        return len(self.calls)
    
    class ChatWrapper:
        """Wraps chat.completions"""
        
        def __init__(self, parent):
            self.parent = parent
            
        @property
        def completions(self):
            """Return completions wrapper"""
            return TokenCountingWrapper.CompletionsWrapper(self.parent)
    
    class CompletionsWrapper:
        """Wraps chat.completions.create"""
        
        def __init__(self, parent):
            self.parent = parent
        
        def create(self, **kwargs):
            """Intercept create call to track usage"""
            start_time = time.time()
            
            # Make actual API call
            response = self.parent.client.chat.completions.create(**kwargs)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract token counts from response.usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            # Record this call
            self.parent.calls.append({
                "model": kwargs.get("model", "unknown"),
                "temperature": kwargs.get("temperature", None),
                "max_tokens": kwargs.get("max_tokens", None),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "timestamp": time.time()
            })
            
            return response
    
    @property
    def chat(self):
        """Return chat wrapper"""
        return self.ChatWrapper(self)
