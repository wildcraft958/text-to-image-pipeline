import time
from typing import Dict, Any
from pipeline.state import PipelineState
from services.vertex_ai import VertexAIService
from services.langfuse_client import LangfuseService
from services.cache_service import CacheService
from utils.ner_utils import NERProcessor
from utils.similarity import SimilarityMatcher
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class PipelineNodes:
    def __init__(self):
        """Initialize all services"""
        try:
            self.vertex_ai = VertexAIService()
            self.langfuse = LangfuseService()
            self.cache = CacheService()
            self.ner_processor = NERProcessor()
            self.similarity_matcher = SimilarityMatcher()
            logger.info("✅ All pipeline services initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipeline services: {e}")
            raise
    
    async def process_input(self, state: PipelineState) -> PipelineState:
        """Process and enhance user input"""
        try:
            logger.info(f"🔄 Processing input for: {state.get('title', 'Unknown')}")
            
            # Extract entities from title and keywords
            combined_text = f"{state['title']} {' '.join(state['keywords'])}"
            entities = self.ner_processor.extract_entities(combined_text, state['keywords'])
            
            # Enhance keywords
            enhanced_keywords = self.ner_processor.enhance_keywords(state['keywords'])
            
            # Create cache key
            cache_key = self.ner_processor.create_cache_key(
                state['title'], 
                enhanced_keywords, 
                entities
            )
            
            # Analyze complexity
            complexity = self.vertex_ai.analyze_prompt_complexity(
                state['title'], 
                enhanced_keywords, 
                state.get('description', '')
            )
            
            # Create new state dict (avoid mutation)
            new_state = dict(state)
            new_state.update({
                'processed_keywords': enhanced_keywords,
                'entities': entities,
                'cache_key': cache_key,
                'prompt_complexity': complexity
            })
            
            logger.info(f"✅ Input processing completed. Cache key: {cache_key}")
            return new_state
            
        except Exception as e:
            logger.error(f"❌ Input processing failed: {e}")
            new_state = dict(state)
            new_state['error'] = f"Input processing failed: {str(e)}"
            return new_state
    
    async def check_cache(self, state: PipelineState) -> PipelineState:
        """Check if similar prompt exists in cache"""
        try:
            logger.info(f"🔍 Checking cache for key: {state.get('cache_key', 'None')}")
            
            # First, try exact cache key match
            cached_prompt = self.cache.get_cached_prompt(state['cache_key'])
            
            new_state = dict(state)
            
            if cached_prompt:
                new_state.update({
                    'cached_prompt': cached_prompt,
                    'used_cache': True
                })
                logger.info("✅ Found exact cache match")
                return new_state
            
            # If no exact match, try semantic similarity
            query_text = f"{state['title']} {' '.join(state['processed_keywords'])}"
            similar_prompts = self.cache.get_similar_prompts()
            
            similar_prompt = self.similarity_matcher.find_similar_prompts(
                query_text, 
                similar_prompts, 
                threshold=settings.SIMILARITY_THRESHOLD
            )
            
            if similar_prompt:
                new_state.update({
                    'cached_prompt': similar_prompt,
                    'used_cache': True
                })
                logger.info("✅ Found similar cache match")
            else:
                new_state['used_cache'] = False
                logger.info("ℹ️ No cache match found")
            
            return new_state
            
        except Exception as e:
            logger.error(f"❌ Cache check failed: {e}")
            new_state = dict(state)
            new_state.update({
                'error': f"Cache check failed: {str(e)}", 
                'used_cache': False
            })
            return new_state
    
    async def generate_prompt(self, state: PipelineState) -> PipelineState:
        """Generate new prompt using LLM with enhanced fallback"""
        try:
            logger.info(f"🔄 Generating prompt for: {state['title']}")
            
            new_state = dict(state)
            
            # Try Langfuse template first
            prompt_template = None
            try:
                prompt_template = self.langfuse.get_prompt_template("image-prompt-generator")
                logger.info("✅ Retrieved Langfuse template")
            except Exception as e:
                logger.warning(f"⚠️ Langfuse template failed: {e}")
            
            generated_prompt = None
            
            if prompt_template:
                try:
                    # Use Langfuse template with variables
                    compiled_prompt = prompt_template.compile(
                        title=state['title'],
                        keywords=', '.join(state['processed_keywords']),
                        style="social-media",
                        complexity=state['prompt_complexity'],
                        photography_style="high-quality, professional",
                        lighting="natural, well-lit",
                        artistic_style="modern, clean"
                    )
                    
                    # Generate using Vertex AI
                    generated_prompt = await self.vertex_ai.generate_prompt(
                        compiled_prompt,
                        state['processed_keywords'],
                        state.get('description', ''),
                        state['prompt_complexity']
                    )
                    logger.info("✅ Generated prompt using Langfuse template")
                    
                except Exception as template_error:
                    logger.warning(f"⚠️ Template compilation failed: {template_error}")
                    generated_prompt = None
            
            # Fallback: Direct generation or simple prompt construction
            if not generated_prompt:
                try:
                    logger.info("🔄 Using fallback prompt generation...")
                    generated_prompt = await self.vertex_ai.generate_prompt(
                        state['title'],
                        state['processed_keywords'], 
                        state.get('description', ''),
                        state['prompt_complexity']
                    )
                    logger.info("✅ Generated prompt using direct method")
                    
                except Exception as vertex_error:
                    logger.warning(f"⚠️ Vertex AI generation failed: {vertex_error}")
                    
                    # Ultimate fallback: Create a simple enhanced prompt
                    title = state['title']
                    keywords = ', '.join(state['processed_keywords'])
                    description = state.get('description', '')
                    
                    generated_prompt = f"Create a high-quality, detailed image of: {title}. "
                    generated_prompt += f"Include these elements: {keywords}. "
                    if description:
                        generated_prompt += f"Additional context: {description}. "
                    generated_prompt += "Style: professional, high-resolution, visually appealing, social media ready."
                    
                    logger.info("✅ Generated fallback prompt")
            
            if generated_prompt:
                new_state['generated_prompt'] = generated_prompt
                logger.info(f"📝 Final prompt: {generated_prompt[:100]}...")
            else:
                error_msg = "All prompt generation methods failed"
                logger.error(f"❌ {error_msg}")
                new_state['error'] = error_msg
            
            return new_state
            
        except Exception as e:
            logger.error(f"❌ Prompt generation failed: {e}")
            new_state = dict(state)
            new_state['error'] = f"Prompt generation failed: {str(e)}"
            return new_state
    
    async def generate_image(self, state: PipelineState) -> PipelineState:
        """Generate image using the prompt with enhanced debugging"""
        try:
            # Debug: Check what prompts are available
            cached_prompt = state.get('cached_prompt')
            generated_prompt = state.get('generated_prompt')
            
            logger.info(f"🔍 Debug - Cached prompt: {'✅' if cached_prompt else '❌'}")
            logger.info(f"🔍 Debug - Generated prompt: {'✅' if generated_prompt else '❌'}")
            
            if cached_prompt:
                logger.info(f"📝 Using cached prompt: {cached_prompt[:50]}...")
            if generated_prompt:
                logger.info(f"📝 Using generated prompt: {generated_prompt[:50]}...")
            
            # Use cached prompt if available, otherwise generated prompt
            prompt_to_use = cached_prompt or generated_prompt
            
            new_state = dict(state)
            
            if not prompt_to_use:
                error_msg = "No prompt available for image generation"
                logger.error(f"❌ {error_msg}")
                logger.error(f"🔍 State keys: {list(state.keys())}")
                new_state['error'] = error_msg
                return new_state
            
            logger.info(f"🎨 Generating image with prompt: {prompt_to_use[:100]}...")
            
            # Generate image
            image_result = await self.vertex_ai.generate_image(
                prompt_to_use, 
                state.get('prompt_complexity', 'simple')
            )
            
            if image_result.get('success'):
                new_state.update({
                    'image_data': image_result['image_data'],
                    'image_url': image_result['image_url']
                })
                logger.info("✅ Image generated successfully")
            else:
                error_msg = f"Image generation failed: {image_result.get('error', 'Unknown error')}"
                logger.error(f"❌ {error_msg}")
                new_state['error'] = error_msg
                
            return new_state
                
        except Exception as e:
            logger.error(f"❌ Image generation error: {e}")
            new_state = dict(state)
            new_state['error'] = f"Image generation failed: {str(e)}"
            return new_state
    
    async def update_cache(self, state: PipelineState) -> PipelineState:
        """Update cache with new prompt if generation was successful"""
        try:
            # Only cache if we generated a new prompt and image was successful
            if (not state.get('used_cache', True) and 
                state.get('generated_prompt') and 
                state.get('image_data')):
                
                original_input = f"{state['title']} {' '.join(state['keywords'])}"
                
                self.cache.cache_prompt(
                    state['cache_key'],
                    original_input,
                    state['generated_prompt'],
                    state['entities']
                )
                logger.info("✅ Cached new prompt")
            
            return state
            
        except Exception as e:
            logger.warning(f"Cache update failed: {e}")
            return state
    
    async def finalize_response(self, state: PipelineState) -> PipelineState:
        """Finalize the response and log to Langfuse"""
        try:
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - state.get('start_time', end_time)
            
            # Log to Langfuse
            try:
                self.langfuse.log_generation(
                    user_input={
                        'user_id': state['user_id'],
                        'title': state['title'],
                        'keywords': state['keywords']
                    },
                    generated_prompt=state.get('cached_prompt') or state.get('generated_prompt', ''),
                    image_success=bool(state.get('image_data')),
                    processing_time=processing_time,
                    used_cache=state.get('used_cache', False)
                )
                logger.info("✅ Logged to Langfuse")
            except Exception as langfuse_error:
                logger.warning(f"Langfuse logging failed: {langfuse_error}")
            
            new_state = dict(state)
            new_state['processing_time'] = processing_time
            return new_state
            
        except Exception as e:
            logger.warning(f"Response finalization failed: {e}")
            return state

# Helper functions for conditional routing
def should_use_cache(state: PipelineState) -> str:
    """Determine if we should use cached prompt or generate new one"""
    if state.get('cached_prompt'):
        return "generate_image"
    return "generate_prompt"

def has_error(state: PipelineState) -> str:
    """Check if there's an error in the state"""
    if state.get('error'):
        return "END"
    return "continue"
