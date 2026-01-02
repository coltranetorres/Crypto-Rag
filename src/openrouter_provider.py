"""
Custom OpenRouter Model Provider for Strands
Supports both LLM completions and embeddings
"""
from typing import List, Dict, Any, Optional, AsyncIterator, Union, AsyncIterable, Type, TypeVar, AsyncGenerator
from openai import OpenAI
import os
from dataclasses import dataclass
from pydantic import BaseModel
from strands.models import Model
from strands.types.content import Message
from strands.types.tools import ToolSpec
from strands.types.streaming import StreamEvent
from src.config.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T', bound=BaseModel)

# Define base classes since they don't exist in Strands
@dataclass
class ModelResponse:
    """Response from model completion"""
    content: str
    model: str
    usage: Dict[str, Any]
    raw_response: Any = None

@dataclass
class EmbeddingResponse:
    """Response from embedding generation"""
    embeddings: List[List[float]]
    model: str
    usage: Dict[str, Any]
    raw_response: Any = None

class ModelProvider:
    """
    Base class for model providers
    """
    pass

class OpenRouterProvider(ModelProvider):
    """
    Custom Strands model provider for OpenRouter
    Handles both chat completions and embeddings
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        site_url: str = "http://localhost:3000",
        site_name: str = "RAG Observability Project",
        default_llm_model: str = "google/gemini-2.5-flash-lite-preview-09-2025",
        default_embedding_model: str = "thenlper/gte-base"
    ):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.getenv("OPENROUTER_API_KEY")
        )
        
        self.extra_headers = {
            "HTTP-Referer": site_url,
            "X-Title": site_name
        }
        
        self.default_llm_model = default_llm_model
        self.default_embedding_model = default_embedding_model
        
        # For tracking
        self.llm_calls = 0
        self.embedding_calls = 0
        self.total_tokens = 0
        
        logger.info(f"OpenRouter Provider initialized: LLM={default_llm_model}, Embeddings={default_embedding_model}")
    
    def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500,
        **kwargs
    ) -> ModelResponse:
        """
        Chat completion using OpenRouter
        """
        model = model or self.default_llm_model
        
        try:
            response = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            self.llm_calls += 1
            self.total_tokens += response.usage.total_tokens
            
            return ModelResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"OpenRouter completion error: {e}", exc_info=True)
            raise
    
    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Generate embeddings using OpenRouter
        """
        model = model or self.default_embedding_model
        
        try:
            response = self.client.embeddings.create(
                extra_headers=self.extra_headers,
                model=model,
                input=texts,
                encoding_format="float",
                **kwargs
            )
            
            self.embedding_calls += 1
            
            embeddings = [item.embedding for item in response.data]
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=model,
                usage={
                    "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else 0
                },
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"OpenRouter embedding error: {e}", exc_info=True)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "llm_calls": self.llm_calls,
            "embedding_calls": self.embedding_calls,
            "total_tokens": self.total_tokens,
            "llm_model": self.default_llm_model,
            "embedding_model": self.default_embedding_model
        }


# Helper wrapper for embeddings
class OpenRouterEmbeddings:
    """
    Simplified embedding wrapper compatible with ChromaDB ingestion
    """
    def __init__(
        self,
        provider: OpenRouterProvider = None,
        model_name: str = "thenlper/gte-base"
    ):
        self.provider = provider or OpenRouterProvider(
            default_embedding_model=model_name
        )
        self.model_name = model_name
        
        # Dimension mapping
        self.dimensions = {
            #"thenlper/gte-small": 384,
            "thenlper/gte-base": 768,
            #"thenlper/gte-large": 1024
        }
        self.dimension = self.dimensions.get(model_name, 768)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        response = self.provider.embed(texts)
        return response.embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        response = self.provider.embed([text])
        return response.embeddings[0]


# Strands Model wrapper for OpenRouterProvider
class OpenRouterModel(Model):
    """
    Strands-compatible Model wrapper for OpenRouterProvider
    Properly implements the Strands Model interface per documentation
    """
    def __init__(self, provider: OpenRouterProvider):
        super().__init__()
        self.provider = provider
        self.model_id = provider.default_llm_model
        self._config = {
            "model": self.model_id,
            "provider": "openrouter",
            "temperature": 0.1,
            "max_tokens": 500
        }
    
    def _format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert Strands Messages format to OpenAI format"""
        formatted = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted.append({
                    "role": msg.get("role", "user"),
                    "content": str(msg.get("content", ""))
                })
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                # Handle Strands Message object
                role = getattr(msg, 'role', 'user')
                content = getattr(msg, 'content', '')
                if isinstance(content, list):
                    # Content might be a list of content blocks
                    content_str = " ".join([str(c) for c in content])
                else:
                    content_str = str(content)
                formatted.append({
                    "role": role,
                    "content": content_str
                })
            else:
                formatted.append({
                    "role": "user",
                    "content": str(msg)
                })
        return formatted
    
    async def stream(
        self,
        messages: List[Message],
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        system_prompt_content: Optional[List] = None,
        **kwargs: Any
    ) -> AsyncIterable[StreamEvent]:
        """
        Stream method required by Strands Model interface
        Yields StreamEvent objects per Strands documentation
        """
        # Format messages
        formatted_messages = self._format_messages(messages)
        
        # Add system prompt if provided
        if system_prompt:
            formatted_messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })
        elif system_prompt_content:
            # Handle system prompt content blocks
            system_content = " ".join([str(block) for block in system_prompt_content])
            formatted_messages.insert(0, {
                "role": "system",
                "content": system_content
            })
        
        # Handle tool specs (convert to OpenAI tools format if needed)
        openai_tools = None
        if tool_specs:
            openai_tools = []
            for tool_spec in tool_specs:
                if isinstance(tool_spec, dict):
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_spec.get("name", ""),
                            "description": tool_spec.get("description", ""),
                            "parameters": tool_spec.get("parameters", {})
                        }
                    })
                elif hasattr(tool_spec, 'name'):
                    # Handle ToolSpec object
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": getattr(tool_spec, 'name', ''),
                            "description": getattr(tool_spec, 'description', ''),
                            "parameters": getattr(tool_spec, 'parameters', {})
                        }
                    })
        
        # Get config parameters
        temperature = kwargs.get("temperature", self._config.get("temperature", 0.1))
        max_tokens = kwargs.get("max_tokens", self._config.get("max_tokens", 500))
        model = kwargs.get("model", self.model_id)
        
        try:
            # Yield message start event
            yield {
                "type": "modelMessageStartEvent",
                "role": "assistant"
            }
            
            # Yield content block start event
            yield {
                "type": "modelContentBlockStartEvent"
            }
            
            # Call OpenRouter API (non-streaming for now, can be enhanced)
            response = self.provider.complete(
                messages=formatted_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=openai_tools,
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens", "model"]}
            )
            
            # Yield content deltas (simulate streaming by chunking)
            content = response.content or ""
            if content:
                chunk_size = 10  # Small chunks for streaming effect
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i + chunk_size]
                    yield {
                        "type": "modelContentBlockDeltaEvent",
                        "delta": {
                            "type": "textDelta",
                            "text": chunk
                        }
                    }
            else:
                # If no content, yield at least one empty delta to signal completion
                yield {
                    "type": "modelContentBlockDeltaEvent",
                    "delta": {
                        "type": "textDelta",
                        "text": ""
                    }
                }
            
            # Yield content block stop event
            yield {
                "type": "modelContentBlockStopEvent"
            }
            
            # Yield message stop event with metadata
            yield {
                "type": "modelMessageStopEvent",
                "stopReason": "endTurn"
            }
            
            # Yield metadata event
            yield {
                "type": "modelMetadataEvent",
                "usage": {
                    "inputTokens": response.usage.get("prompt_tokens", 0),
                    "outputTokens": response.usage.get("completion_tokens", 0),
                    "totalTokens": response.usage.get("total_tokens", 0)
                },
                "metrics": {
                    "latencyMs": 0  # Could track actual latency
                }
            }
            
        except Exception as e:
            # Yield error in stop event
            yield {
                "type": "modelMessageStopEvent",
                "stopReason": "error",
                "error": str(e)
            }
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self._config.copy()
    
    def update_config(self, **kwargs: Any) -> None:
        """Update model configuration"""
        self._config.update(kwargs)
        if "model" in kwargs:
            self.model_id = kwargs["model"]
    
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: List[Message],
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Union[T, Any]], None]:
        """
        Get structured output using tool calling.
        Implements the abstract method from Model base class.
        """
        try:
            # Convert Pydantic model to tool specification
            # This is a simplified version - in production you'd use proper conversion
            tool_spec = {
                "name": output_model.__name__.lower(),
                "description": output_model.__doc__ or f"Extract {output_model.__name__}",
                "parameters": output_model.model_json_schema() if hasattr(output_model, 'model_json_schema') else {}
            }
            
            # Use the stream method with tool specification
            response_events = []
            async for event in self.stream(
                messages=prompt,
                tool_specs=[tool_spec],
                system_prompt=system_prompt,
                **kwargs
            ):
                response_events.append(event)
            
            # Extract the last stop event to get the response
            stop_event = None
            for event in reversed(response_events):
                if event.get("type") == "modelMessageStopEvent":
                    stop_event = event
                    break
            
            # For now, we'll use a simple approach: call the model and parse JSON
            # In a full implementation, you'd extract tool use from the stream
            formatted_messages = self._format_messages(prompt)
            if system_prompt:
                formatted_messages.insert(0, {"role": "system", "content": system_prompt})
            
            # Add instruction to return JSON
            formatted_messages.append({
                "role": "user",
                "content": f"Return the response as a JSON object matching this schema: {output_model.model_json_schema()}"
            })
            
            response = self.provider.complete(
                messages=formatted_messages,
                model=self.model_id,
                **kwargs
            )
            
            # Try to parse JSON from response
            import json
            try:
                # Try to extract JSON from the response
                content = response.content
                # Look for JSON in the response
                if "{" in content and "}" in content:
                    json_start = content.find("{")
                    json_end = content.rfind("}") + 1
                    json_str = content[json_start:json_end]
                    data = json.loads(json_str)
                    parsed_output = output_model(**data)
                    yield {"output": parsed_output}
                else:
                    raise ValueError("No JSON found in response")
            except Exception as e:
                raise ValueError(f"Failed to parse structured output: {e}")
                
        except Exception as e:
            yield {"error": str(e)}
            raise

