from langfuse import Langfuse
from typing import Optional, Dict, Any
from config.settings import settings

class LangfuseService:
    def __init__(self):
        """Initialize Langfuse client"""
        self.client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST
        )
        
        # Initialize prompt templates
        self._setup_prompt_templates()
    
    def _setup_prompt_templates(self):
        """Setup initial prompt templates in Langfuse"""
        try:
            # Create base prompt template for image generation
            self.client.create_prompt(
                name="image-prompt-generator",
                type="text",
                prompt="""You are an expert prompt engineer for AI image generation. 
                Create a detailed, visually rich prompt for: {{title}}
                
                Keywords to include: {{keywords}}
                Style preference: {{style}}
                Complexity: {{complexity}}
                
                Include appropriate modifiers:
                - Photography: {{photography_style}}
                - Lighting: {{lighting}}
                - Artistic style: {{artistic_style}}
                
                Generate a concise but descriptive prompt under 500 characters:""",
                labels=["production"],
                config={
                    "model": settings.LLM_MODEL,
                    "temperature": 0.7,
                    "max_tokens": 200
                }
            )
            
            # Create style templates
            styles = [
                ("minimalist", "clean, simple, elegant, white background"),
                ("vibrant", "bright colors, high contrast, energetic"),
                ("professional", "corporate, clean, sophisticated"),
                ("artistic", "creative, unique perspective, artistic flair"),
                ("social-media", "instagram-worthy, engaging, trendy")
            ]
            
            for style_name, style_desc in styles:
                self.client.create_prompt(
                    name=f"style-{style_name}",
                    type="text",
                    prompt=style_desc,
                    labels=["production"],
                    tags=["style"]
                )
                
        except Exception as e:
            print(f"Warning: Could not setup Langfuse templates: {e}")
    
    def get_prompt_template(self, template_name: str = "image-prompt-generator") -> Optional[Any]:
        """Get prompt template from Langfuse"""
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
        """Log the generation process to Langfuse"""
        try:
            trace = self.client.trace(
                name="text-to-image-generation",
                user_id=user_input.get("user_id"),
                metadata={
                    "used_cache": used_cache,
                    "processing_time": processing_time,
                    "image_success": image_success
                }
            )
            
            # Log prompt generation
            trace.generation(
                name="prompt-generation",
                model=settings.LLM_MODEL,
                input=user_input,
                output=generated_prompt,
                metadata={"used_cache": used_cache}
            )
            
            # Log image generation
            trace.generation(
                name="image-generation", 
                model=settings.IMAGE_MODEL,
                input=generated_prompt,
                output={"success": image_success},
                metadata={"processing_time": processing_time}
            )
            
        except Exception as e:
            print(f"Could not log to Langfuse: {e}")
