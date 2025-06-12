# ai-service/app/core/llm_client.py (SIMPLIFIED VERSION)
import logging
import asyncio
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from anthropic import AsyncAnthropic

from app.config.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Simplified LLM client for Gemini and Claude"""

    def __init__(self):
        # Configure Gemini
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)
        else:
            self.gemini_model = None

        # Configure Claude
        if settings.CLAUDE_API_KEY:
            self.claude_client = AsyncAnthropic(api_key=settings.CLAUDE_API_KEY)
        else:
            self.claude_client = None

    async def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        use_fallback: bool = False,
    ) -> str:
        """Generate text completion using primary LLM with fallback"""

        max_tokens = max_tokens or settings.MAX_TOKENS
        temperature = temperature or settings.TEMPERATURE

        # Determine which LLM to use
        primary_llm = settings.FALLBACK_LLM if use_fallback else settings.PRIMARY_LLM

        try:
            if primary_llm == "gemini" and self.gemini_model:
                return await self._generate_gemini_completion(
                    prompt, system_prompt, max_tokens, temperature
                )
            elif primary_llm == "claude" and self.claude_client:
                return await self._generate_claude_completion(
                    prompt, system_prompt, max_tokens, temperature
                )
            else:
                # Try the other LLM if primary not available
                if not use_fallback:
                    return await self.generate_completion(
                        prompt,
                        system_prompt,
                        max_tokens,
                        temperature,
                        use_fallback=True,
                    )
                else:
                    return "I apologize, but I don't have access to any LLM services at the moment. Please check your API key configuration."

        except Exception as e:
            logger.error(f"Error with {primary_llm}: {str(e)}")

            # Try fallback if not already using it
            if not use_fallback:
                logger.info(f"Trying fallback LLM: {settings.FALLBACK_LLM}")
                return await self.generate_completion(
                    prompt, system_prompt, max_tokens, temperature, use_fallback=True
                )
            else:
                return f"I encountered an error while processing your request: {str(e)}"

    async def _generate_gemini_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
    ) -> str:
        """Generate completion using Gemini"""
        try:
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}"

            # Configure generation
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )

            # Generate response
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                full_prompt,
                generation_config=generation_config,
            )

            return response.text

        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise

    async def _generate_claude_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
    ) -> str:
        """Generate completion using Claude"""
        try:
            messages = [{"role": "user", "content": prompt}]

            response = await self.claude_client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "You are a helpful AI assistant.",
                messages=messages,
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            raise


# Global client instance
llm_client = LLMClient()
