"""
LLM Client for iAgent and MemRec
Supports Azure OpenAI API and OpenAI-compatible APIs (TogetherAI, Anyscale, vLLM, etc.)
"""
import os
import json
import threading
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
from openai import AzureOpenAI, OpenAI

from src.models.response_cache import ResponseCache, request_key

GEMINI_OPENAI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Transient errors worth retrying (rate limit, overload, server error, timeout)
RETRYABLE_MARKERS = ("429", "rate_limit", "ratelimit", "resource_exhausted", "500", "502", "503", "504",
                     "unavailable", "overloaded", "internal", "timeout", "timed out", "connection")


class CallBudgetExceeded(RuntimeError):
    """Raised when a run tries to make more physical LLM calls than allowed"""


class LLMClient:
    """LLM client supporting multiple providers"""

    def __init__(
        self,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        model: str = "gpt-4o-mini",
        provider_name: str = "azure_openai",  # "azure_openai", "openai" or "gemini"
        save_conversations: bool = False,
        conversation_log_path: Optional[str] = None,
        cache_path: Optional[str] = None,  # SQLite response cache (shared across runs)
        max_physical_calls: Optional[int] = None,  # Hard cap on non-cached calls
        extra_body: Optional[Dict] = None  # e.g. {"chat_template_kwargs": {"enable_thinking": False}} for Qwen3 on vLLM
    ):
        """
        Initialize LLM client
        
        Args:
            api_endpoint: API endpoint (Azure OpenAI or OpenAI-compatible)
            api_key: API key
            api_version: API version (for Azure OpenAI)
            model: Model name
            provider_name: "azure_openai" or "openai" (for OpenAI-compatible APIs)
            save_conversations: Whether to save conversation history
            conversation_log_path: Path to save conversation logs (JSONL format)
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.model = model
        self.provider_name = provider_name
        self.extra_body = extra_body
        
        # Get from env if not provided
        if not self.api_endpoint:
            if provider_name == "azure_openai":
                self.api_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            elif provider_name == "gemini":
                self.api_endpoint = GEMINI_OPENAI_ENDPOINT
            elif provider_name == "local":  # self-hosted OpenAI-compatible server (vLLM)
                self.api_endpoint = os.getenv('LOCAL_LLM_BASE', 'http://127.0.0.1:8000/v1')
            else:
                self.api_endpoint = os.getenv('OPENAI_API_BASE') or os.getenv('OPENAI_BASE_URL')

        if not self.api_key:
            if provider_name == "azure_openai":
                self.api_key = os.getenv('AZURE_OPENAI_API_KEY')
            elif provider_name == "gemini":
                self.api_key = os.getenv('GEMINI_API_KEY')
            elif provider_name == "local":
                self.api_key = os.getenv('LOCAL_LLM_KEY', 'EMPTY')
            else:
                self.api_key = os.getenv('OPENAI_API_KEY') or os.getenv('TOGETHER_API_KEY') or os.getenv('ANYSCALE_API_KEY')
        
        if not self.api_endpoint or not self.api_key:
            raise ValueError(
                f"API credentials not provided for {provider_name}. "
                f"Please set endpoint and api_key, or use environment variables."
            )
        
        # Initialize client based on provider
        if provider_name == "azure_openai":
            self.client = AzureOpenAI(
                azure_endpoint=self.api_endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
        else:  # OpenAI-compatible (Gemini, TogetherAI, Anyscale, vLLM, etc.)
            self.client = OpenAI(
                base_url=self.api_endpoint,
                api_key=self.api_key,
                timeout=120.0
            )

        # Token usage tracking (physical = calls that reached the API)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.physical_calls = 0
        self.cache_hits = 0
        # Logical = what the run would cost without the cache (cache hits counted with their recorded usage)
        self.logical_input_tokens = 0
        self.logical_output_tokens = 0
        self.max_physical_calls = max_physical_calls
        self._stats_lock = threading.Lock()
        self.cache = ResponseCache(cache_path) if cache_path else None
        
        # Conversation logging
        self.save_conversations = save_conversations
        self.conversation_log_path = conversation_log_path
        self.conversation_count = 0
        
        if self.save_conversations and self.conversation_log_path:
            log_path = Path(self.conversation_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"💬 Conversation logging enabled: {log_path}")
    
    def _log_conversation(
        self,
        messages: List[Dict[str, str]],
        response: str,
        metadata: Optional[Dict] = None
    ):
        """Log a conversation to file (JSONL format)"""
        if not self.save_conversations or not self.conversation_log_path:
            return
        
        self.conversation_count += 1
        
        log_entry = {
            'id': self.conversation_count,
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'messages': messages,
            'response': response,
            'metadata': metadata or {}
        }
        
        try:
            with open(self.conversation_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Warning: Failed to log conversation: {e}")
    
    def _request_kwargs(self, messages, temperature, max_tokens, json_schema) -> Dict:
        """Build the chat.completions kwargs (also the cache key payload)"""
        kwargs = {
            'model': self.model,
            'messages': messages
        }
        
        # Use max_completion_tokens for gpt-5-nano, max_tokens for others
        # gpt-5-nano requires max_completion_tokens instead of max_tokens
        if 'gpt-5-nano' in self.model.lower() or 'nano' in self.model.lower():
            kwargs['max_completion_tokens'] = max_tokens
            # gpt-5-nano does NOT support custom temperature (only default 1.0)
        else:
            kwargs['max_tokens'] = max_tokens
            # Only add temperature if specified (for non-nano models)
            if temperature is not None:
                kwargs['temperature'] = temperature
        
        # Add JSON schema if provided
        if json_schema:
            kwargs['response_format'] = {
                'type': 'json_schema',
                'json_schema': json_schema
            }
        if self.extra_body:
            kwargs['extra_body'] = self.extra_body
        return kwargs
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = 0.7,
        max_tokens: int = 4000,
        json_schema: Optional[Dict] = None,
        max_retries: int = 5
    ) -> str:
        """
        Generate response from LLM with exponential backoff retry
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (None = use API default ~1.0)
            max_tokens: Maximum tokens to generate (increased to 4000 to avoid truncation)
            json_schema: Optional JSON schema for structured output
            max_retries: Maximum number of retries for rate limit errors
            
        Returns:
            Generated text
        """
        import time
        
        kwargs = self._request_kwargs(messages, temperature, max_tokens, json_schema)
        
        # Cache lookup: identical request -> stored response, no API call
        key = None
        if self.cache is not None:
            key = request_key(kwargs)
            entry = self.cache.get_entry(key)
            if entry is not None:
                with self._stats_lock:
                    self.cache_hits += 1
                    self.logical_input_tokens += entry[1]
                    self.logical_output_tokens += entry[2]
                self._log_conversation(messages=messages, response=entry[0], metadata={
                    'cached': True, 'usage': {'prompt_tokens': entry[1], 'completion_tokens': entry[2]}})
                return entry[0]
        
        # Hard budget guard on physical calls
        with self._stats_lock:
            if self.max_physical_calls is not None and self.physical_calls >= self.max_physical_calls:
                raise CallBudgetExceeded(f"physical call budget {self.max_physical_calls} reached")
            self.physical_calls += 1
        
        # Retry with exponential backoff for transient errors (rate limit, overload, 5xx, timeout)
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(**kwargs)
                choice = completion.choices[0] if completion.choices else None
                message = getattr(choice, 'message', None)
                response = getattr(message, 'content', None)
                if not response:
                    # Gemini occasionally returns a choice without message/content; retry it
                    raise RuntimeError(f"empty response, finish_reason={getattr(choice, 'finish_reason', None)} (unavailable)")
                
                prompt_tokens = completion_tokens = 0
                usage = getattr(completion, 'usage', None)
                if usage:
                    prompt_tokens = getattr(usage, 'prompt_tokens', None) or getattr(usage, 'input_tokens', 0) or 0
                    completion_tokens = getattr(usage, 'completion_tokens', None) or getattr(usage, 'output_tokens', 0) or 0
                with self._stats_lock:
                    self.total_input_tokens += prompt_tokens
                    self.total_output_tokens += completion_tokens
                    self.total_requests += 1
                    self.logical_input_tokens += prompt_tokens
                    self.logical_output_tokens += completion_tokens
                
                if self.cache is not None:
                    self.cache.put(key, self.model, response, prompt_tokens, completion_tokens)
                
                # Log conversation if enabled
                self._log_conversation(
                    messages=messages,
                    response=response,
                    metadata={
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                        'has_json_schema': json_schema is not None,
                        'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}
                    }
                )
                
                return response
            except Exception as e:
                error_str = str(e).lower()
                retryable = any(m in error_str for m in RETRYABLE_MARKERS)
                if retryable and attempt < max_retries - 1:
                    # Exponential backoff: 2s, 4s, 8s, ... capped at 60s
                    wait_time = min(60, 2 ** (attempt + 1))
                    print(f"Transient LLM error (attempt {attempt + 1}/{max_retries}): {str(e)[:120]}; waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                print(f"Error calling LLM: {str(e)[:300]}")
                raise
        
        # Should not reach here, but just in case
        raise Exception(f"Failed after {max_retries} attempts")
    
    def generate_json(
        self,
        messages: List[Dict[str, str]],
        properties: Dict[str, Dict],
        temperature: Optional[float] = 0.7,
        max_tokens: int = 4000,
        debug_logger = None
    ) -> Dict:
        """
        Generate JSON response from LLM
        
        Args:
            messages: List of message dicts
            properties: JSON schema properties
            temperature: Sampling temperature
            max_tokens: Maximum tokens (increased to 4000 to avoid truncation)
            debug_logger: Optional function to log debug info
            
        Returns:
            Parsed JSON dict
        """
        # Debug log: record complete LLM input
        if debug_logger:
            debug_logger("\n>>> LLM INPUT (Messages) <<<")
            for i, msg in enumerate(messages, 1):
                debug_logger(f"[Message {i}] Role: {msg.get('role', 'unknown')}")
                content = msg.get('content', '')
                # If content is too long, show preview
                if len(content) > 2000:
                    debug_logger(f"Content (first 2000 chars):\n{content[:2000]}\n... (truncated, total {len(content)} chars)")
                else:
                    debug_logger(f"Content:\n{content}")
                debug_logger("")  # Empty line separator
            
            debug_logger(">>> LLM Schema (Properties) <<<")
            debug_logger(json.dumps(properties, indent=2, ensure_ascii=False))
        
        # Build JSON schema
        json_schema = {
            'name': 'response',
            'strict': True,
            'schema': {
                'type': 'object',
                'properties': properties,
                'required': list(properties.keys()),
                'additionalProperties': False
            }
        }
        
        response_text = self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema
        )
        
        try:
            response_dict = json.loads(response_text)
            
            # Debug log: record LLM output
            if debug_logger:
                debug_logger("\n>>> LLM OUTPUT (Parsed JSON) <<<")
                debug_logger(json.dumps(response_dict, indent=2, ensure_ascii=False))
                debug_logger("")  # Empty line separator
            
            return response_dict
        except json.JSONDecodeError as e:
            # Never keep a malformed response in the cache
            if self.cache is not None:
                self.cache.delete(request_key(self._request_kwargs(messages, temperature, max_tokens, json_schema)))
            print(f"Error parsing JSON: {e}")
            print(f"Raw response: {response_text}")
            if debug_logger:
                debug_logger(f"\n❌ JSON Parse Error: {e}")
                debug_logger(f"Raw response:\n{response_text}")
            raise
    
    def get_token_stats(self) -> Dict[str, int]:
        """Get cumulative token usage statistics"""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'total_requests': self.total_requests,
            'physical_calls': self.physical_calls,
            'cache_hits': self.cache_hits,
            'logical_input_tokens': self.logical_input_tokens,
            'logical_output_tokens': self.logical_output_tokens,
            'avg_input_tokens': self.total_input_tokens / self.total_requests if self.total_requests > 0 else 0,
            'avg_output_tokens': self.total_output_tokens / self.total_requests if self.total_requests > 0 else 0,
        }
    
    def reset_token_stats(self):
        """Reset token usage statistics"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
