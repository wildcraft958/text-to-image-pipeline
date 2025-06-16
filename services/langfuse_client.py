"""
langfuse_service.py - Compatible with Langfuse Python SDK v3 (OTEL-based)
Maintains backward compatibility with existing code while supporting the latest SDK.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

# Import handling for different SDK versions
try:
    from langfuse import Langfuse, get_client
except ImportError:
    from langfuse.client import Langfuse  # type: ignore
    get_client = None

from config.settings import settings

logger = logging.getLogger(__name__)


class LangfuseService:
    """
    Updated Langfuse service compatible with SDK v3 (OTEL-based) while maintaining
    backward compatibility with existing code.
    """

    def __init__(self) -> None:
        self.client: Optional[Langfuse] = None
        self._is_v3_sdk = False

        # Check if credentials are available
        if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
            logger.warning("⚠️ Langfuse credentials not provided - service will be disabled")
            return

        try:
            # Try to initialize with v3 SDK pattern first
            if get_client:
                self.client = get_client()
                self._is_v3_sdk = True
                logger.info("✅ Initialized Langfuse SDK v3 (OTEL-based)")
            else:
                # Fallback to v2 SDK pattern
                self.client = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST,
                )
                self._is_v3_sdk = False
                logger.info("✅ Initialized Langfuse SDK v2")

            # Test the connection with appropriate method
            self._test_connection()
            logger.info("✅ Langfuse connection verified")

        except Exception as exc:
            logger.warning(f"⚠️ Langfuse connection failed: {exc}")
            logger.warning("   Pipeline will continue without Langfuse integration")
            self.client = None

    def _test_connection(self) -> None:
        """Test connection using the appropriate method for the SDK version."""
        if not self.client:
            return

        # Try different connection test methods
        test_methods = ["auth_check", "check_auth", "ping"]
        for method_name in test_methods:
            if hasattr(self.client, method_name):
                getattr(self.client, method_name)()
                return
        
        # If no test method found, assume connection is OK
        logger.debug("No connection test method found, assuming connection is OK")

    def get_prompt_template(self, template_name: str = "image-prompt-generator") -> Optional[Any]:
        """Get prompt template from Langfuse - compatible with both SDK versions."""
        if not self.client:
            logger.warning("Langfuse client not available")
            return None
            
        try:
            # Try different method names for getting prompts
            if hasattr(self.client, 'get_prompt'):
                prompt = self.client.get_prompt(template_name)
            elif hasattr(self.client, 'prompt'):
                prompt = self.client.prompt(template_name)
            else:
                logger.warning(f"No prompt retrieval method found in SDK")
                return None
                
            logger.info(f"✅ Retrieved prompt template: {template_name}")
            return prompt
        except Exception as e:
            logger.warning(f"Could not fetch prompt template {template_name}: {e}")
            return None

    def log_generation(self,
                      user_input: Dict[str, Any],
                      generated_prompt: str,
                      image_success: bool,
                      processing_time: float,
                      used_cache: bool) -> None:
        """
        Log the generation process to Langfuse - updated for SDK v3 compatibility.
        """
        if not self.client:
            logger.debug("Langfuse client not available, skipping logging")
            return

        try:
            if self._is_v3_sdk:
                self._log_generation_v3(user_input, generated_prompt, image_success, processing_time, used_cache)
            else:
                self._log_generation_v2(user_input, generated_prompt, image_success, processing_time, used_cache)
                
            logger.info("✅ Successfully logged to Langfuse")

        except Exception as e:
            logger.warning(f"Could not log to Langfuse: {e}")

    def _log_generation_v3(self, user_input: Dict[str, Any], generated_prompt: str, 
                          image_success: bool, processing_time: float, used_cache: bool) -> None:
        """Log using SDK v3 (OTEL-based) methods."""
        # Create trace using v3 API
        with self.client.start_as_current_span(
            name="text-to-image-generation",
            input=user_input,
            metadata={
                "used_cache": used_cache,
                "processing_time": processing_time,
                "image_success": image_success
            }
        ) as trace:
            
            # Set trace-level attributes
            if hasattr(trace, 'update_trace'):
                trace.update_trace(
                    user_id=user_input.get("user_id"),
                    tags=["text-to-image", "generation"]
                )

            # Log prompt generation step
            with trace.start_as_current_generation(
                name="prompt-generation",
                model=settings.LLM_MODEL,
                input=user_input,
                output=generated_prompt,
                metadata={"used_cache": used_cache}
            ) as prompt_gen:
                pass

            # Log image generation step
            with trace.start_as_current_generation(
                name="image-generation",
                model=settings.IMAGE_MODEL,
                input=generated_prompt,
                output={"success": image_success},
                metadata={"processing_time": processing_time}
            ) as image_gen:
                pass

            # Update final trace output
            if hasattr(trace, 'update'):
                trace.update(
                    output={
                        "success": image_success,
                        "prompt_used": generated_prompt,
                        "processing_time": processing_time
                    }
                )

    def _log_generation_v2(self, user_input: Dict[str, Any], generated_prompt: str,
                          image_success: bool, processing_time: float, used_cache: bool) -> None:
        """Log using SDK v2 methods (fallback)."""
        # Create trace using v2 API
        trace = self.client.trace(
            name="text-to-image-generation",
            input=user_input,
            user_id=user_input.get("user_id"),
            metadata={
                "used_cache": used_cache,
                "processing_time": processing_time,
                "image_success": image_success
            }
        )

        # Log prompt generation step
        prompt_generation = trace.generation(
            name="prompt-generation",
            model=settings.LLM_MODEL,
            input=user_input,
            output=generated_prompt,
            metadata={"used_cache": used_cache}
        )

        # Log image generation step
        image_generation = trace.generation(
            name="image-generation", 
            model=settings.IMAGE_MODEL,
            input=generated_prompt,
            output={"success": image_success},
            metadata={"processing_time": processing_time}
        )

        # Update final trace output
        trace.update(
            output={
                "success": image_success,
                "prompt_used": generated_prompt,
                "processing_time": processing_time
            }
        )

    def flush(self) -> None:
        """Manually flush events to Langfuse - important for short-lived applications."""
        if self.client and hasattr(self.client, 'flush'):
            try:
                self.client.flush()
                logger.debug("✅ Flushed events to Langfuse")
            except Exception as e:
                logger.warning(f"Could not flush to Langfuse: {e}")

    def is_available(self) -> bool:
        """Check if Langfuse service is available."""
        return self.client is not None

    def __del__(self):
        """Ensure events are flushed when the service is destroyed."""
        self.flush()
