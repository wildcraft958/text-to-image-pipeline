import time
from typing import Dict, Any
from pipeline.state import PipelineState
from services.vertex_ai import VertexAIService
from services.langfuse_client import LangfuseService
from services.cache_service import CacheService
from utils.ner_utils import NERProcessor
from utils.similarity import SimilarityMatcher
from config.settings import settings

class PipelineNodes:
    def __init__(self):
        """Initialize all services"""
        self.vertex_ai = VertexAIService()
        self.langfuse = LangfuseService()
        self.cache = CacheService()
        self.ner_processor = NERProcessor()
        self.similarity_matcher = SimilarityMatcher()
    
    async def process_input(self, state: PipelineState) -> PipelineState:
        """Process and enhance user input"""
        try:
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
            
            return {
                **state,
                'processed_keywords': enhanced_keywords,
                'entities': entities,
                'cache_key': cache_key,
                'prompt_complexity': complexity
            }
            
        except Exception as e:
            return {**state, 'error': f"Input processing failed: {str(e)}"}
    
    async def check_cache(self, state: PipelineState) -> PipelineState:
        """Check if similar prompt exists in cache"""
        try:
            # First, try exact cache key match
            cached_prompt = self.cache.get_cached_prompt(state['cache_key'])
            
            if cached_prompt:
                return {
                    **state, 
                    'cached_prompt': cached_prompt,
                    'used_cache': True
                }
            
            # If no exact match, try semantic similarity
            query_text = f"{state['title']} {' '.join(state['processed_keywords'])}"
            similar_prompts = self.cache.get_similar_prompts()
            
            similar_prompt = self.similarity_matcher.find_similar_prompts(
                query_text, 
                similar_prompts, 
                threshold=settings.SIMILARITY_THRESHOLD
            )
            
            if similar_prompt:
                return {
                    **state, 
                    'cached_prompt': similar_prompt,
                    'used_cache': True
                }
            
            return {**state, 'used_cache': False}
            
        except Exception as e:
            return {**state, 'error': f"Cache check failed: {str(e)}", 'used_cache': False}
    
    async def generate_prompt(self, state: PipelineState) -> PipelineState:
        """Generate new prompt using LLM"""
        try:
            # Get prompt template from Langfuse
            prompt_template = self.langfuse.get_prompt_template()
            
            if prompt_template:
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
            else:
                # Fallback to direct generation
                generated_prompt = await self.vertex_ai.generate_prompt(
                    state['title'],
                    state['processed_keywords'], 
                    state.get('description', ''),
                    state['prompt_complexity']
                )
            
            return {**state, 'generated_prompt': generated_prompt}
            
        except Exception as e:
            return {**state, 'error': f"Prompt generation failed: {str(e)}"}
    
    async def generate_image(self, state: PipelineState) -> PipelineState:
        """Generate image using the prompt"""
        try:
            # Use cached prompt if available, otherwise generated prompt
            prompt_to_use = state.get('cached_prompt') or state.get('generated_prompt')
            
            if not prompt_to_use:
                return {**state, 'error': "No prompt available for image generation"}
            
            # Generate image
            image_result = await self.vertex_ai.generate_image(
                prompt_to_use, 
                state['prompt_complexity']
            )
            
            if image_result['success']:
                return {
                    **state,
                    'image_data': image_result['image_data'],
                    'image_url': image_result['image_url']
                }
            else:
                return {**state, 'error': "Image generation failed"}
                
        except Exception as e:
            return {**state, 'error': f"Image generation failed: {str(e)}"}
    
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
            
            return state
            
        except Exception as e:
            print(f"Cache update failed: {e}")
            return state
    
    async def finalize_response(self, state: PipelineState) -> PipelineState:
        """Finalize the response and log to Langfuse"""
        try:
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - state.get('start_time', end_time)
            
            # Log to Langfuse
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
            
            return {**state, 'processing_time': processing_time}
            
        except Exception as e:
            print(f"Response finalization failed: {e}")
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
