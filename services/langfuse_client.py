from langfuse import get_client, observe
from typing import Optional, Dict, Any
from config.settings import settings
import logging

class LangfuseService:
    def __init__(self):
        """Initialize Langfuse client using SDK v3"""
        try:
            # Initialize the global client - SDK v3 pattern
            self.client = get_client()
            
            # Verify connection
            self.client.auth_check()
            print("✅ Langfuse connection verified")
            
        except Exception as e:
            print(f"⚠️ Langfuse connection failed: {e}")
            self.client = None
    
    def get_prompt_template(self, template_name: str = "image-prompt-generator") -> Optional[Any]:
        """Get prompt template from Langfuse"""
        if not self.client:
            return None
            
        try:
            return self.client.get_prompt(template_name)
        except Exception as e:
            print(f"Could not fetch prompt template {template_name}: {e}")
            return None
    
    def log_generation(self, 
                      user_input: Dict[str, Any], 
                      generated_prompt: str, 
                      image_success: bool,
                      processing_time: float,
                      used_cache: bool):
        """Log the generation process to Langfuse using SDK v3"""
        if not self.client:
            print("Langfuse client not available, skipping logging")
            return
            
        try:
            # Create a trace for the text-to-image generation
            with self.client.start_as_current_span(
                name="text-to-image-generation",
                input=user_input
            ) as root_span:
                
                # Update trace with user metadata
                root_span.update_trace(
                    user_id=user_input.get("user_id"),
                    metadata={
                        "used_cache": used_cache,
                        "processing_time": processing_time,
                        "image_success": image_success
                    }
                )
                
                # Log prompt generation step
                with self.client.start_as_current_generation(
                    name="prompt-generation",
                    model=settings.LLM_MODEL,
                    input=user_input
                ) as prompt_gen:
                    prompt_gen.update(
                        output=generated_prompt,
                        metadata={"used_cache": used_cache}
                    )
                
                # Log image generation step
                with self.client.start_as_current_generation(
                    name="image-generation",
                    model=settings.IMAGE_MODEL,
                    input=generated_prompt
                ) as image_gen:
                    image_gen.update(
                        output={"success": image_success},
                        metadata={"processing_time": processing_time}
                    )
                
                # Update final trace output
                root_span.update_trace(
                    output={
                        "success": image_success,
                        "prompt_used": generated_prompt,
                        "processing_time": processing_time
                    }
                )
                
        except Exception as e:
            print(f"Could not log to Langfuse: {e}")
