# Updated langfuse_client.py - Fixed SDK v3 initialization and error handling

from langfuse import Langfuse
from typing import Optional, Dict, Any
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class LangfuseService:
    def __init__(self):
        """Initialize Langfuse client using SDK v3 - FIXED initialization"""
        self.client = None
        
        try:
            # Check if credentials are available
            if not settings.LANGFUSE_PUBLIC_KEY or not settings.LANGFUSE_SECRET_KEY:
                logger.warning("⚠️ Langfuse credentials not provided - service will be disabled")
                return
            
            # Initialize the Langfuse client - SDK v3 pattern
            self.client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST
            )
            
            # Test the connection
            self.client.auth_check()
            logger.info("✅ Langfuse connection verified")
            
        except Exception as e:
            logger.warning(f"⚠️ Langfuse connection failed: {e}")
            logger.warning("   Pipeline will continue without Langfuse integration")
            self.client = None

    def get_prompt_template(self, template_name: str = "image-prompt-generator") -> Optional[Any]:
        """Get prompt template from Langfuse - FIXED error handling"""
        if not self.client:
            logger.warning("Langfuse client not available")
            return None
            
        try:
            prompt = self.client.get_prompt(template_name)
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
                      used_cache: bool):
        """Log the generation process to Langfuse - FIXED logging pattern"""
        if not self.client:
            logger.debug("Langfuse client not available, skipping logging")
            return

        try:
            # Create a trace for the text-to-image generation
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
            prompt_generation = self.client.generation(
                name="prompt-generation",
                model=settings.LLM_MODEL,
                input=user_input,
                output=generated_prompt,
                metadata={"used_cache": used_cache},
                trace_id=trace.id
            )

            # Log image generation step
            image_generation = self.client.generation(
                name="image-generation", 
                model=settings.IMAGE_MODEL,
                input=generated_prompt,
                output={"success": image_success},
                metadata={"processing_time": processing_time},
                trace_id=trace.id
            )

            # Update final trace output
            trace.update(
                output={
                    "success": image_success,
                    "prompt_used": generated_prompt,
                    "processing_time": processing_time
                }
            )

            logger.info("✅ Successfully logged to Langfuse")

        except Exception as e:
            logger.warning(f"Could not log to Langfuse: {e}")

    def is_available(self) -> bool:
        """Check if Langfuse service is available"""
        return self.client is not None