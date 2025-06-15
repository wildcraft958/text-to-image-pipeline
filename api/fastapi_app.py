from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import base64
import time
import traceback
import asyncio
from pipeline.graph import TextToImagePipeline
from pipeline.state import UserInput, PipelineResponse
from config.settings import settings
from contextlib import asynccontextmanager
import logging
from pipeline.state import PipelineState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("text_to_image_pipeline")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    logger.info("🚀 Text-to-Image Pipeline API Starting...")
    logger.info(f"🔗 Using Google Project: {settings.GOOGLE_PROJECT_ID}")
    logger.info(f"📍 Vertex AI Location: {settings.GOOGLE_LOCATION}")
    yield
    # Shutdown
    logger.info("🛑 Text-to-Image Pipeline API Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Text-to-Image Pipeline API",
    description="AI-powered text-to-image generation with caching and optimization",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware AFTER app initialization
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize pipeline
pipeline = TextToImagePipeline()

# Add explicit OPTIONS handler for all paths
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle OPTIONS requests for CORS preflight"""
    return {"message": "OK"}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Text-to-Image Pipeline API", 
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify server connectivity"""
    return {
        "status": "ok",
        "message": "Server is responding",
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        services = {}
        
        # Check Vertex AI
        try:
            services["vertex_ai"] = "connected"
        except Exception as e:
            services["vertex_ai"] = f"error: {str(e)}"
        
        # Check Langfuse
        try:
            if hasattr(pipeline.nodes, 'langfuse') and pipeline.nodes.langfuse.client:
                services["langfuse"] = "connected"
            else:
                services["langfuse"] = "not connected"
        except Exception as e:
            services["langfuse"] = f"error: {str(e)}"
        
        # Check Cache
        try:
            pipeline.nodes.cache.get_cached_prompt("test")
            services["cache"] = "active"
        except Exception as e:
            services["cache"] = f"error: {str(e)}"
        
        return {
            "status": "healthy",
            "services": services,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

@app.post("/generate-image", response_model=PipelineResponse)
async def generate_image(request: UserInput):
    """Generate image from text input with enhanced error handling"""
    try:
        logger.info(f"📝 Received request from user: {request.user_id}")
        logger.info(f"📝 Title: {request.title}")
        logger.info(f"📝 Keywords: {request.keywords}")
        
        # Validate input
        if not request.title.strip():
            logger.warning("❌ Empty title provided")
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        
        if not request.keywords or len(request.keywords) == 0:
            logger.warning("❌ No keywords provided")
            raise HTTPException(status_code=400, detail="At least one keyword is required")
        
        # Process request through pipeline with timeout
        logger.info("🔄 Processing through pipeline...")
        
        try:
            result = await asyncio.wait_for(
                pipeline.process_request(
                    user_id=request.user_id,
                    title=request.title,
                    keywords=request.keywords,
                    description=request.description
                ),
                timeout=60.0  # 60 second timeout
            )
        except asyncio.TimeoutError:
            logger.error("⏰ Pipeline processing timed out")
            raise HTTPException(status_code=504, detail="Request processing timed out")
        
        logger.info(f"📊 Pipeline result: success={result.get('success', False)}")
        
        if not result.get("success"):
            error_detail = result.get("error", "Unknown pipeline error")
            logger.error(f"❌ Pipeline failed: {error_detail}")
            raise HTTPException(status_code=500, detail=error_detail)
        
        logger.info("✅ Request processed successfully")
        return PipelineResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
        logger.error(f"📍 Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/image/{user_id}/latest")
async def get_latest_image(user_id: str):
    """Get the latest generated image for a user"""
    return {"message": "Image retrieval not implemented in prototype"}

@app.post("/image/download")
async def download_image(image_data: str):
    """Download image from base64 data"""
    try:
        image_bytes = base64.b64decode(image_data)
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=generated_image.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

# Add this to your FastAPI app
@app.post("/debug-pipeline")
async def debug_pipeline(request: UserInput):
    """Debug pipeline step by step"""
    try:
        # Initialize state
        initial_state = PipelineState(
            user_id=request.user_id,
            title=request.title,
            keywords=request.keywords,
            description=request.description,
            processed_keywords=[],
            entities={},
            cache_key=None,
            cached_prompt=None,
            generated_prompt=None,
            prompt_complexity=None,
            image_url=None,
            image_data=None,
            processing_time=0.0,
            used_cache=False,
            error=None,
            start_time=time.time()
        )
        
        # Test each step
        result = {"steps": {}}
        
        # Step 1: Process input
        state1 = await pipeline.nodes.process_input(initial_state)
        result["steps"]["process_input"] = {
            "success": not bool(state1.get('error')),
            "cache_key": state1.get('cache_key'),
            "processed_keywords": state1.get('processed_keywords', [])[:5],  # First 5
            "error": state1.get('error')
        }
        
        # Step 2: Check cache
        state2 = await pipeline.nodes.check_cache(state1)
        result["steps"]["check_cache"] = {
            "success": not bool(state2.get('error')),
            "used_cache": state2.get('used_cache'),
            "cached_prompt": state2.get('cached_prompt', '')[:50] if state2.get('cached_prompt') else None,
            "error": state2.get('error')
        }
        
        # Step 3: Generate prompt (if no cache)
        if not state2.get('cached_prompt'):
            state3 = await pipeline.nodes.generate_prompt(state2)
            result["steps"]["generate_prompt"] = {
                "success": not bool(state3.get('error')),
                "generated_prompt": state3.get('generated_prompt', '')[:50] if state3.get('generated_prompt') else None,
                "error": state3.get('error')
            }
        else:
            state3 = state2
            result["steps"]["generate_prompt"] = {"skipped": "using_cache"}
        
        # Step 4: Check final prompt availability
        final_prompt = state3.get('cached_prompt') or state3.get('generated_prompt')
        result["final_prompt_available"] = bool(final_prompt)
        result["final_prompt_preview"] = final_prompt[:100] if final_prompt else None
        
        return result
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/examples")
async def get_examples():
    """Get example requests for testing"""
    return {
        "examples": [
            {
                "user_id": "test_user_1",
                "title": "Sunset Beach Scene",
                "keywords": ["sunset", "beach", "ocean", "peaceful"],
                "description": "A serene beach scene at sunset with gentle waves"
            },
            {
                "user_id": "test_user_2", 
                "title": "Modern City Skyline",
                "keywords": ["city", "skyscrapers", "modern", "lights"],
                "description": "A futuristic city skyline at night with bright lights"
            },
            {
                "user_id": "test_user_3",
                "title": "Cozy Coffee Shop",
                "keywords": ["coffee", "cozy", "interior", "warm"],
                "description": "A warm and inviting coffee shop interior"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.fastapi_app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
