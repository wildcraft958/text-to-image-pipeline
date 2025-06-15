import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.preview.vision_models import ImageGenerationModel
# from google.cloud import aiplatform
from typing import Dict, Any
import json
from google.oauth2 import service_account
# import base64
import io
# from PIL import Image
from config.settings import settings

class VertexAIService:
    def __init__(self):
        """Initialize Vertex AI services"""
        vertexai.init(
            project=settings.GOOGLE_PROJECT_ID,
            location=settings.GOOGLE_LOCATION,
            credentials=service_account.Credentials.from_service_account_info(json.load(open(settings.GOOGLE_APPLICATION_CREDENTIALS)))
        )
        
        # Initialize models
        self.llm_model = GenerativeModel(settings.LLM_MODEL)
        self.image_model = ImageGenerationModel.from_pretrained(settings.IMAGE_MODEL)
    
    async def generate_prompt(self, 
                            title: str, 
                            keywords: list, 
                            description: str = None,
                            complexity: str = "simple") -> str:
        """Generate an enhanced prompt using Gemini"""
        
        # Base prompt template with advanced modifiers
        base_template = """
        You are an expert prompt engineer for AI image generation. Create a detailed, visually rich prompt 
        that will generate high-quality images suitable for social media.
        
        Title: {title}
        Keywords: {keywords}
        Description: {description}
        Complexity Level: {complexity}
        
        Instructions:
        1. Create a vivid, detailed prompt incorporating the title and keywords
        2. Add appropriate artistic modifiers based on complexity:
           - For simple: Basic lighting, composition, and style
           - For complex: Advanced camera settings, artistic styles, materials, and techniques
        3. Include social media optimization elements (vibrant colors, engaging composition)
        4. Ensure the prompt is under 500 characters for optimal performance
        
        Generate only the image prompt, no explanations:
        """
        
        prompt = base_template.format(
            title=title,
            keywords=", ".join(keywords),
            description=description or "No additional description",
            complexity=complexity
        )
        
        try:
            response = await self.llm_model.generate_content_async(prompt)
            return response.text.strip()
        except Exception as e:
            raise Exception(f"Failed to generate prompt: {str(e)}")
    
    async def generate_image(self, prompt: str, complexity: str = "simple") -> Dict[str, Any]:
        """Generate image using Imagen 3"""
        
        # Adjust parameters based on complexity
        if complexity == "complex":
            guidance_scale = 15
            number_of_images = 1
        else:
            guidance_scale = 10
            number_of_images = 1
        
        try:
            response = self.image_model.generate_images(
                prompt=prompt,
                number_of_images=number_of_images,
                guidance_scale=guidance_scale,
                aspect_ratio="1:1",
                safety_filter_level="block_some",
                person_generation="dont_allow"
            )
            
            # Convert to bytes for storage/transmission
            image = response.images[0]
            img_bytes = io.BytesIO()
            image._pil_image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            return {
                "image_data": img_bytes.getvalue(),
                "image_url": None,  # In production, upload to cloud storage
                "success": True
            }
            
        except Exception as e:
            raise Exception(f"Failed to generate image: {str(e)}")
    
    def analyze_prompt_complexity(self, title: str, keywords: list, description: str = None) -> str:
        """Simple heuristic to determine prompt complexity"""
        total_length = len(title) + len(" ".join(keywords)) + len(description or "")
        keyword_count = len(keywords)
        
        if total_length > 100 or keyword_count > 5:
            return "complex"
        return "simple"
