from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import base64
from pipeline.graph import TextToImagePipeline
from pipeline.state import UserInput, PipelineResponse
from config.settings import settings

# Initialize FastAPI app
app = FastAPI(
    title="Text-to-Image Pipeline API",
    description="AI-powered text-to-image generation with caching and optimization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = TextToImagePipeline()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Text-to-Image Pipeline API Starting...")
    print(f"🔗 Using Google Project: {settings.GOOGLE_PROJECT_ID}")
    print(f"📍 Vertex AI Location: {settings.GOOGLE_LOCATION}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Text-to-Image Pipeline API", 
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "vertex_ai": "connected",
            "langfuse": "connected", 
            "cache": "active"
        }
    }

@app.post("/generate-image", response_model=PipelineResponse)
async def generate_image(request: UserInput):
    """Generate image from text input"""
    try:
        # Validate input
        if not request.title.strip():
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        
        if not request.keywords or len(request.keywords) == 0:
            raise HTTPException(status_code=400, detail="At least one keyword is required")
        
        # Process request through pipeline
        result = await pipeline.process_request(
            user_id=request.user_id,
            title=request.title,
            keywords=request.keywords,
            description=request.description
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
        return PipelineResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/image/{user_id}/latest")
async def get_latest_image(user_id: str):
    """Get the latest generated image for a user"""
    # This would typically fetch from a database or cloud storage
    # For now, return a placeholder response
    return {"message": "Image retrieval not implemented in prototype"}

@app.post("/image/download")
async def download_image(image_data: str):
    """Download image from base64 data"""
    try:
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=generated_image.png"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

# Example usage endpoints for testing
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
