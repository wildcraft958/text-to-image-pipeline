from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict
from pydantic import BaseModel

class PipelineState(TypedDict):
    """State object for the text-to-image pipeline"""
    # Input
    user_id: str
    title: str
    keywords: List[str]
    description: Optional[str]
    
    # Processing
    processed_keywords: List[str]
    entities: Dict[str, List[str]]
    cache_key: Optional[str]
    cached_prompt: Optional[str]
    
    # LLM Generation
    generated_prompt: Optional[str]
    prompt_complexity: Optional[str]  # "simple" or "complex"
    
    # Image Generation
    image_url: Optional[str]
    image_data: Optional[bytes]
    
    # Metadata
    processing_time: float
    used_cache: bool
    error: Optional[str]

class UserInput(BaseModel):
    """Input model for API requests"""
    user_id: str
    title: str
    keywords: List[str]
    description: Optional[str] = None

class PipelineResponse(BaseModel):
    """Response model for the pipeline"""
    success: bool
    image_url: Optional[str] = None
    prompt_used: Optional[str] = None
    processing_time: float
    used_cache: bool
    error: Optional[str] = None
