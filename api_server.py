"""
FastAPI server for habit image processing API.
Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
Or: python api_server.py
"""

from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional
import sys
import os
import json

# Import the processing function from the existing module
# Handle filename with spaces by using importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "habit_processor",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "API for cropping and prediction.py")
)
habit_processor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(habit_processor)
process_habit_image = habit_processor.process_habit_image

app = FastAPI(
    title="bordi.ai Image Processing API",
    description="API for processing habit tracking images with cropping and classification",
    version="1.0.0"
)

# Add CORS middleware to allow requests from any origin
# In production, you may want to restrict this to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - change this in production!
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class ProcessImageRequest(BaseModel):
    """Request model for image processing"""
    user_id: str
    image_url: HttpUrl  # Validates that image_url is a valid URL
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "f9370f47-ae09-4b29-a42b-44aef42d5389",
                "image_url": "https://res.cloudinary.com/db6fegsqa/image/upload/v1763427064/wbzlfsefitkddzqzv5y2.jpg"
            }
        }


class ProcessImageResponse(BaseModel):
    """Response model for image processing"""
    image_url: str
    timestamp: str
    average_confidence: float
    total_true: int
    total_false: int
    results: list
    cropped_images_directory: str


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "bordi.ai Image Processing API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/process-image", response_model=ProcessImageResponse)
async def process_image(
    request: Request,
    json_body: Optional[ProcessImageRequest] = None,
    user_id: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None)
):
    """
    Process a habit tracking image by cropping cells and classifying them.
    
    Accepts JSON body, form data, or raw JSON string:
    - **user_id**: The user ID to look up crop values in image_crop_values table
    - **image_url**: The URL of the image to process
    
    Returns classification results, statistics, and timestamp.
    """
    try:
        final_user_id = None
        final_image_url = None
        
        # Try to get from Pydantic model (proper JSON with Content-Type)
        if json_body:
            final_user_id = json_body.user_id
            final_image_url = str(json_body.image_url)
        # Try form data
        elif user_id and image_url:
            final_user_id = user_id
            final_image_url = image_url
        else:
            # Try to parse raw body as JSON (for n8n without proper Content-Type)
            try:
                body = await request.json()
                if "user_id" in body and "image_url" in body:
                    final_user_id = body["user_id"]
                    final_image_url = body["image_url"]
            except:
                # If JSON parsing fails, try reading as bytes and parsing
                try:
                    body_bytes = await request.body()
                    body_str = body_bytes.decode('utf-8')
                    body = json.loads(body_str)
                    if "user_id" in body and "image_url" in body:
                        final_user_id = body["user_id"]
                        final_image_url = body["image_url"]
                except:
                    pass
        
        if not final_user_id or not final_image_url:
            raise HTTPException(
                status_code=422, 
                detail="user_id and image_url are required in the request body"
            )
        
        result = process_habit_image(final_user_id, final_image_url)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/process-image")
async def process_image_get(user_id: str, image_url: str):
    """
    GET endpoint for image processing (alternative to POST).
    Useful for testing or simple integrations.
    """
    try:
        result = process_habit_image(user_id, image_url)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting API server on http://{host}:{port}")
    print(f"API documentation available at http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)

