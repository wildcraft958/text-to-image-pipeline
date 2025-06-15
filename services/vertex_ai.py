import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel
from typing import Dict, Any
import json
from google.oauth2 import service_account
import io
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class VertexAIService:
    def __init__(self):
        """Initialize Vertex AI services"""
        try:
            vertexai.init(
                project=settings.GOOGLE_PROJECT_ID,
                location=settings.GOOGLE_LOCATION,
                credentials=service_account.Credentials.from_service_account_info(
                    json.load(open(settings.GOOGLE_APPLICATION_CREDENTIALS))
                )
            )
            
            # Initialize models with error handling
            try:
                self.llm_model = GenerativeModel(settings.LLM_MODEL)
                logger.info(f"✅ LLM Model initialized: {settings.LLM_MODEL}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize LLM model: {e}")
                self.llm_model = None
            
            try:
                self.image_model = ImageGenerationModel.from_pretrained(settings.IMAGE_MODEL)
                logger.info(f"✅ Image Model initialized: {settings.IMAGE_MODEL}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Image model: {e}")
                self.image_model = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vertex AI: {e}")
            raise
    
    async def generate_prompt(self, 
                            title: str, 
                            keywords: list, 
                            description: str = None,
                            complexity: str = "simple") -> str:
        """Generate an enhanced prompt using Gemini with fallback"""
        
        if not self.llm_model:
            logger.warning("LLM model not available, using fallback prompt generation")
            return self._create_fallback_prompt(title, keywords, description, complexity)
        
        # Enhanced prompt template
        base_template = f"""
        Create a detailed, visually rich prompt for AI image generation.
        
        Title: {title}
        Keywords: {', '.join(keywords)}
        Description: {description or 'No additional description'}
        Style: {complexity} complexity for social media
        
        Generate a concise but descriptive image prompt (under 400 characters) that includes:
        - The main subject: {title}
        - Key elements: {', '.join(keywords)}
        - Artistic style: high-quality, professional, social media ready
        - Lighting and composition details
        
        Prompt:
        """
        
        try:
            response = await self.llm_model.generate_content_async(base_template)
            if response and response.text:
                generated_prompt = response.text.strip()
                logger.info(f"✅ LLM generated prompt: {generated_prompt[:50]}...")
                return generated_prompt
            else:
                logger.warning("Empty response from LLM, using fallback")
                return self._create_fallback_prompt(title, keywords, description, complexity)
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._create_fallback_prompt(title, keywords, description, complexity)
    
    def _create_fallback_prompt(self, title: str, keywords: list, description: str, complexity: str) -> str:
        """Create a fallback prompt when LLM is unavailable"""
        prompt = f"Create a high-quality, detailed image of: {title}. "
        prompt += f"Include these elements: {', '.join(keywords)}. "
        if description:
            prompt += f"Additional context: {description}. "
        
        # Add complexity-based modifiers
        if complexity == "complex":
            prompt += "Style: professional photography, cinematic lighting, high resolution, artistic composition, vibrant colors, social media ready."
        else:
            prompt += "Style: clean, professional, well-lit, high-quality, social media ready."
        
        return prompt
    
    async def generate_image(self, prompt: str, complexity: str = "simple") -> Dict[str, Any]:
        """Generate image using Imagen with enhanced error handling"""
        
        if not self.image_model:
            return {
                "image_data": None,
                "image_url": None,
                "success": False,
                "error": "Image model not available"
            }
        
        # Adjust parameters based on complexity
        if complexity == "complex":
            guidance_scale = 15
            number_of_images = 1
        else:
            guidance_scale = 10
            number_of_images = 1
        
        try:
            logger.info(f"🎨 Generating image with Imagen model...")
            
            response = self.image_model.generate_images(
                prompt=prompt,
                number_of_images=number_of_images,
                guidance_scale=guidance_scale,
                aspect_ratio="1:1",
                safety_filter_level="block_some",
                person_generation="allow_adult"  # Changed from "dont_allow"
            )
            
            # Enhanced error handling for response
            if not response or not hasattr(response, 'images') or not response.images:
                logger.error("❌ Empty or invalid response from Imagen")
                return {
                    "image_data": None,
                    "image_url": None,
                    "success": False,
                    "error": "No images generated by Imagen model"
                }
            
            if len(response.images) == 0:
                logger.error("❌ Response contains empty images list")
                return {
                    "image_data": None,
                    "image_url": None,
                    "success": False,
                    "error": "Images list is empty"
                }
            
            # Safely access the first image
            try:
                image = response.images[0]
                if not hasattr(image, '_pil_image') or image._pil_image is None:
                    logger.error("❌ Invalid image object returned")
                    return {
                        "image_data": None,
                        "image_url": None,
                        "success": False,
                        "error": "Invalid image object"
                    }
                
                # Convert to bytes for storage/transmission
                img_bytes = io.BytesIO()
                image._pil_image.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                
                logger.info("✅ Image generated successfully")
                return {
                    "image_data": img_bytes.getvalue(),
                    "image_url": None,  # In production, upload to cloud storage
                    "success": True
                }
                
            except IndexError as e:
                logger.error(f"❌ Index error accessing image: {e}")
                return {
                    "image_data": None,
                    "image_url": None,
                    "success": False,
                    "error": f"Failed to access generated image: {str(e)}"
                }
            
        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            return {
                "image_data": None,
                "image_url": None,
                "success": False,
                "error": f"Image generation error: {str(e)}"
            }
    
    def analyze_prompt_complexity(self, title: str, keywords: list, description: str = None) -> str:
        """Simple heuristic to determine prompt complexity"""
        total_length = len(title) + len(" ".join(keywords)) + len(description or "")
        keyword_count = len(keywords)
        
        if total_length > 100 or keyword_count > 5:
            return "complex"
        return "simple"
