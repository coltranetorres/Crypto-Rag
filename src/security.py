"""
Security utilities for input validation and attack detection
"""
import re
from typing import Tuple, Optional, List
from datetime import datetime
from src.config.logging_config import get_logger

logger = get_logger(__name__)

# Known jailbreak patterns
JAILBREAK_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)",
    r"(?i)disregard\s+(all\s+)?(previous|above|prior|your\s+programming)",
    r"(?i)forget\s+(everything|all|your)\s+(above|previous|instructions?)",
    r"(?i)you\s+are\s+now\s+(in\s+)?(\w+\s+)?mode",
    r"(?i)pretend\s+(you\s+are|to\s+be|you're)",
    r"(?i)act\s+as\s+(if\s+you\s+are|a|an|if)",
    r"(?i)roleplay\s+as",
    r"(?i)jailbreak",
    r"(?i)DAN\s*(\d+)?",
    r"(?i)do\s+anything\s+now",
    r"(?i)developer\s+mode",
    r"(?i)unrestricted\s+mode",
    r"(?i)bypass\s+(safety|filter|restriction)",
    r"(?i)enable\s+.*mode",
    r"(?i)\[system\]|\[assistant\]|\[user\]",  # Fake message delimiters
    r"(?i)<\|.*\|>",  # Special token attempts
    r"(?i)###\s*(instruction|system|human|assistant)",
]

# Prompt leaking patterns
LEAKING_PATTERNS = [
    r"(?i)what\s+(is|are)\s+your\s+(system\s+)?prompt",
    r"(?i)show\s+(me\s+)?your\s+(system\s+)?instructions?",
    r"(?i)reveal\s+(your\s+)?(system\s+)?(prompt|original)",
    r"(?i)print\s+(your\s+)?(system\s+)?(prompt|words|the\s+words)",
    r"(?i)output\s+(your\s+)?initialization",
    r"(?i)repeat\s+(the\s+)?(words\s+)?(above|your\s+initialization)",
    r"(?i)what\s+were\s+you\s+told",
    r"(?i)original\s+(instructions?|prompt)",
    r"(?i)training\s+data",
    r"(?i)how\s+were\s+you\s+(trained|programmed|built)",
]

# Suspicious character patterns
OBFUSCATION_PATTERNS = [
    r"[\u200b-\u200f\u2060-\u206f]",  # Zero-width and invisible characters
    r"[\u0300-\u036f]{3,}",  # Excessive combining diacritical marks
    r"[^\x00-\x7F]{20,}",  # Long non-ASCII sequences (potential encoding tricks)
]


class PatternBasedFilter:
    """
    Fast pattern-based security filter (no LLM calls)
    First line of defense before LLM-based guardrails
    """
    
    def __init__(
        self,
        max_input_length: int = 4000,
        enable_jailbreak_detection: bool = True,
        enable_leaking_detection: bool = True,
        enable_obfuscation_detection: bool = True
    ):
        self.max_input_length = max_input_length
        self.enable_jailbreak_detection = enable_jailbreak_detection
        self.enable_leaking_detection = enable_leaking_detection
        self.enable_obfuscation_detection = enable_obfuscation_detection
        
        # Compile patterns for performance
        self.jailbreak_patterns = [re.compile(p) for p in JAILBREAK_PATTERNS]
        self.leaking_patterns = [re.compile(p) for p in LEAKING_PATTERNS]
        self.obfuscation_patterns = [re.compile(p) for p in OBFUSCATION_PATTERNS]
    
    def check_input(self, user_input: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check input against security patterns
        
        Returns:
            Tuple of (should_block, reason, attack_type)
        """
        # Length check
        if len(user_input) > self.max_input_length:
            logger.warning(f"Input too long: {len(user_input)} chars")
            return True, f"Input exceeds maximum length ({self.max_input_length} chars)", "length_exceeded"
        
        # Empty input
        if not user_input or not user_input.strip():
            return True, "Empty input", "empty_input"
        
        # Jailbreak detection
        if self.enable_jailbreak_detection:
            for pattern in self.jailbreak_patterns:
                if pattern.search(user_input):
                    logger.warning(f"Jailbreak pattern detected: {pattern.pattern[:50]}")
                    return True, "Potential jailbreak attempt detected", "jailbreak"
        
        # Prompt leaking detection
        if self.enable_leaking_detection:
            for pattern in self.leaking_patterns:
                if pattern.search(user_input):
                    logger.warning(f"Leaking pattern detected: {pattern.pattern[:50]}")
                    return True, "Potential prompt extraction attempt", "prompt_leaking"
        
        # Obfuscation detection
        if self.enable_obfuscation_detection:
            for pattern in self.obfuscation_patterns:
                if pattern.search(user_input):
                    logger.warning(f"Obfuscation detected: {pattern.pattern[:30]}")
                    return True, "Suspicious character patterns detected", "obfuscation"
        
        return False, None, None
    
    def sanitize_for_prompt(self, text: str) -> str:
        """
        Sanitize text before inserting into prompts
        Helps prevent injection through delimiters
        """
        # Remove potential delimiter injections
        sanitized = re.sub(r'```+', '`', text)
        sanitized = re.sub(r'---+', '-', sanitized)
        sanitized = re.sub(r'###', '#', sanitized)
        sanitized = re.sub(r'\[/?(?:SYSTEM|USER|ASSISTANT|INST)\]', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized


class SecurityAuditLog:
    """
    Log security events for monitoring and analysis
    """
    
    def __init__(self):
        self.events: List[dict] = []
    
    def log_blocked(
        self,
        user_input: str,
        reason: str,
        attack_type: str,
        filter_stage: str  # "pattern" or "llm"
    ):
        """Log a blocked request"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "action": "blocked",
            "input_preview": user_input[:100] + "..." if len(user_input) > 100 else user_input,
            "reason": reason,
            "attack_type": attack_type,
            "filter_stage": filter_stage
        }
        self.events.append(event)
        logger.warning(f"SECURITY_BLOCK: {attack_type} - {reason}")
    
    def get_recent_events(self, count: int = 50) -> List[dict]:
        """Get recent security events"""
        return self.events[-count:]

