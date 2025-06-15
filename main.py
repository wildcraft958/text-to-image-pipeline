"""
Text-to-Image Pipeline with LangGraph, Langfuse, and Google Vertex AI
A prototype implementation for social media image generation
"""

import asyncio
import os
from dotenv import load_dotenv
from pipeline.graph import TextToImagePipeline
from api.fastapi_app import app
import uvicorn
from config.settings import settings

# Load environment variables
load_dotenv()

async def test_pipeline():
    """Test the pipeline with sample data"""
    print("🧪 Testing Text-to-Image Pipeline...")
    
    pipeline = TextToImagePipeline()
    
    # Test data
    test_cases = [
        {
            "user_id": "test_user_1",
            "title": "Mountain Landscape",
            "keywords": ["mountain", "landscape", "nature", "scenic"],
            "description": "A beautiful mountain landscape with snow-capped peaks"
        },
        {
            "user_id": "test_user_2",
            "title": "Tech Startup Office",
            "keywords": ["office", "modern", "tech", "creative"],
            "description": "A modern tech startup office space"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['title']}")
        
        result = await pipeline.process_request(**test_case)
        
        print(f"✅ Success: {result['success']}")
        print(f"⚡ Processing Time: {result['processing_time']:.2f}s")
        print(f"🔄 Used Cache: {result['used_cache']}")
        
        if result['success']:
            print(f"📝 Prompt: {result['prompt_used'][:100]}...")
            if result['image_data']:
                print(f"🖼️ Image Generated: {len(result['image_data'])} bytes")
        else:
            print(f"❌ Error: {result['error']}")

def setup_environment():
    """Setup environment and check requirements"""
    print("🔧 Setting up environment...")
    
    # Check required environment variables
    required_vars = [
        "GOOGLE_PROJECT_ID",
        "LANGFUSE_PUBLIC_KEY", 
        "LANGFUSE_SECRET_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please set them in your .env file")
        return False
    
    print("✅ Environment setup complete!")
    return True

def main():
    """Main application entry point"""
    print("🚀 Text-to-Image Pipeline Starting...")
    print("=" * 50)
    
    if not setup_environment():
        return
    
    # Choice of operation
    print("\nChoose operation:")
    print("1. Run API server")
    print("2. Test pipeline")
    print("3. Both (test then serve)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "2":
        # Run tests
        asyncio.run(test_pipeline())
    elif choice == "1":
        # Run API server
        print(f"\n🌐 Starting API server on http://{settings.API_HOST}:{settings.API_PORT}")
        print(f"📚 API docs available at http://{settings.API_HOST}:{settings.API_PORT}/docs")
        
        uvicorn.run(
            app,
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=True
        )
    elif choice == "3":
        # Test then serve
        asyncio.run(test_pipeline())
        input("\nPress Enter to start API server...")
        
        print(f"\n🌐 Starting API server on http://{settings.API_HOST}:{settings.API_PORT}")
        uvicorn.run(
            app,
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=True
        )
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
