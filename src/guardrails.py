"""
Guardrails validation module
Extracted from RAGAgent to follow Single Responsibility Principle
"""
from typing import Dict, Any, Optional, Tuple
import os
import yaml
from src.config.logging_config import get_logger
from src.openrouter_provider import OpenRouterProvider
from src.exceptions import GuardrailBlockedError

logger = get_logger(__name__)


class GuardrailsValidator:
    """
    Validates user input and bot responses using LLM-based guardrails
    """
    
    def __init__(
        self,
        provider: OpenRouterProvider,
        guardrails_path: Optional[str] = None
    ):
        """
        Initialize guardrails validator
        
        Args:
            provider: OpenRouterProvider instance for LLM moderation
            guardrails_path: Path to guardrails YAML file
        """
        self.provider = provider
        self.guardrails = self._load_guardrails(guardrails_path)
        
        if self.guardrails:
            logger.info("Guardrails loaded and enabled")
        else:
            logger.warning("Guardrails not loaded - validation disabled")
    
    def _load_guardrails(self, guardrails_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load guardrails from YAML file
        
        Args:
            guardrails_path: Path to guardrails YAML file
            
        Returns:
            Guardrails configuration dictionary or None if not found
        """
        if guardrails_path is None:
            guardrails_path = "guardrails.yaml"
        
        if not os.path.exists(guardrails_path):
            logger.warning(f"Guardrails file not found at {guardrails_path}, continuing without guardrails")
            return None
        
        try:
            with open(guardrails_path, 'r', encoding='utf-8') as f:
                guardrails = yaml.safe_load(f)
            logger.info(f"Guardrails loaded from {guardrails_path}")
            return guardrails
        except Exception as e:
            logger.error(f"Failed to load guardrails: {e}", exc_info=True)
            return None
    
    def _get_prompt(self, section: str, task: str) -> Optional[str]:
        """
        Get a prompt from guardrails by section and task
        
        Args:
            section: Section name (e.g., "input", "output")
            task: Task name (e.g., "self_check_input")
            
        Returns:
            Prompt content or None if not found
        """
        if not self.guardrails:
            return None
        
        section_data = self.guardrails.get(section, {})
        prompts = section_data.get("prompts", [])
        
        for prompt in prompts:
            if prompt.get("task") == task:
                return prompt.get("content", "")
        
        return None
    
    def validate_input(self, user_input: str) -> Tuple[bool, Optional[str]]:
        """
        Check if user input should be blocked using guardrails
        
        Args:
            user_input: User input to validate
            
        Returns:
            Tuple of (should_block, reason) - True if should block, False if allow
        """
        if not self.guardrails:
            return False, None
        
        prompt_template = self._get_prompt("input", "self_check_input")
        if not prompt_template:
            logger.warning("Input guardrail prompt not found in configuration")
            return False, None
        
        # Replace template variable
        prompt = prompt_template.replace("{{ user_input }}", user_input)
        
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = self.provider.complete(
                messages=messages,
                temperature=0.0,
                max_tokens=10
            )
            
            answer = response.content.strip().lower()
            
            # Check if answer indicates blocking
            if "yes" in answer:
                reason = "Input blocked by guardrails"
                logger.info(f"Input blocked: {user_input[:50]}...")
                return True, reason
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"Input guardrail check failed: {e}", exc_info=True)
            # On error, allow the input (fail open)
            return False, None
    
    def validate_output(self, user_input: str, bot_response: str) -> Tuple[bool, Optional[str]]:
        """
        Check if bot response should be blocked using guardrails
        
        Args:
            user_input: Original user input
            bot_response: Bot response to validate
            
        Returns:
            Tuple of (should_block, reason) - True if should block, False if allow
        """
        if not self.guardrails:
            return False, None
        
        prompt_template = self._get_prompt("output", "self_check_output")
        if not prompt_template:
            logger.warning("Output guardrail prompt not found in configuration")
            return False, None
        
        # Replace template variables
        prompt = prompt_template.replace("{{ user_input }}", user_input)
        prompt = prompt.replace("{{ bot_response }}", bot_response)
        
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = self.provider.complete(
                messages=messages,
                temperature=0.0,
                max_tokens=10
            )
            
            answer = response.content.strip().lower()
            
            # Check if answer indicates blocking
            if "yes" in answer:
                reason = "Response blocked by guardrails"
                logger.info(f"Response blocked for input: {user_input[:50]}...")
                return True, reason
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"Output guardrail check failed: {e}", exc_info=True)
            # On error, allow the response (fail open)
            return False, None

