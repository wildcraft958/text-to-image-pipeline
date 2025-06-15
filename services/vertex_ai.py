import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from google import genai
from google.genai import types
from typing import Dict, Any
import json
from google.oauth2 import service_account
import io
import base64
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class VertexAIService:
    def __init__(self):
        """Initialize both Vertex AI and Google AI Studio services"""
        try:
            # Initialize Vertex AI for image generation
            vertexai.init(
                project=settings.GOOGLE_PROJECT_ID,
                location=settings.GOOGLE_LOCATION,
                credentials=service_account.Credentials.from_service_account_info(
                    json.load(open(settings.GOOGLE_APPLICATION_CREDENTIALS))
                )
            )
            
            # Initialize Imagen model
            try:
                self.image_model = ImageGenerationModel.from_pretrained(settings.IMAGE_MODEL)
                logger.info(f"✅ Image Model initialized: {settings.IMAGE_MODEL}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Image model: {e}")
                self.image_model = None
            
            # Initialize Google AI Studio client for Gemma
            try:
                self.gemma_client = genai.Client(api_key=settings.GOOGLE_AI_STUDIO_API_KEY)
                self.gemma_config = types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=400,
                    top_p=0.7,
                    top_k=40
                )
                logger.info(f"✅ Gemma client initialized with model: {settings.LLM_MODEL}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Gemma client: {e}")
                self.gemma_client = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}")
            raise

    async def generate_prompt(self,
                            title: str,
                            keywords: list,
                            description: str = None,
                            complexity: str = "simple") -> str:
        """Generate an enhanced prompt using Gemma via Google AI Studio"""
        if not self.gemma_client:
            logger.warning("Gemma client not available, using fallback prompt generation")
            return self._create_fallback_prompt(title, keywords, description, complexity)

        # Create the prompt template similar to your working example
        prompt_template = f"""
You are an expert visual-prompt engineer. Your job is to convert a simple title plus a list of descriptive tags into a rich, coherent, and 
evocative prompt for a vision model. Be concise, precise, and imaginative.

Title: {title}
Tags: {', '.join(keywords)}
Description: {description or 'No additional description'}
Complexity Level: {complexity}

Guidelines:
1. Capture the core concept implied by the title.
2. Weave in each tag so that it influences the style, mood, color palette, composition, or atmosphere.
3. Describe setting, lighting, textures, and any dramatic details.
4. If appropriate, add brief artistic style references.
5. For {complexity} complexity, {'add rich technical details and artistic modifiers' if complexity == 'complex' else 'keep it clean and professional'}.
6. Do not exceed 300 words.

**Important Guideline**
Output only the final image prompt. Do not preface with explanations or anything extra.

Generate the image prompt now.
        """.strip()

        try:
            response = self.gemma_client.models.generate_content(
                model=settings.LLM_MODEL,  # This should be 'gemma-3n-e4b-it'
                contents=prompt_template,
                config=self.gemma_config
            )
            
            if response and response.text:
                generated_prompt = response.text.strip()
                logger.info(f"✅ Gemma generated prompt: {generated_prompt[:50]}...")
                return generated_prompt
            else:
                logger.warning("Empty response from Gemma, using fallback")
                return self._create_fallback_prompt(title, keywords, description, complexity)
                
        except Exception as e:
            logger.error(f"Gemma generation failed: {e}")
            return self._create_fallback_prompt(title, keywords, description, complexity)

    def _create_fallback_prompt(self, title: str, keywords: list, description: str, complexity: str) -> str:
        """Create a fallback prompt when Gemma is unavailable"""
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
        """Generate image using Imagen - this remains unchanged"""
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
                person_generation="allow_adult"
            )

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
                
                # Convert to base64 for JSON response
                image_b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                
                logger.info("✅ Image generated successfully")
                return {
                    "image_data": image_b64,
                    "image_url": None,
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
