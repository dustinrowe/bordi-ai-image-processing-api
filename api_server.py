"""
FastAPI server for habit image processing API.
Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
Or: python api_server.py
"""

from fastapi import FastAPI, HTTPException, Request
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
async def process_image(request: Request):
    """
    Process a habit tracking image by cropping cells and classifying them.
    
    Accepts JSON body (with or without Content-Type header):
    - **user_id**: The user ID to look up crop values in image_crop_values table
    - **image_url**: The URL of the image to process
    
    Returns classification results, statistics, and timestamp.
    """
    try:
        # Read the raw body first (before FastAPI tries to parse it)
        body_bytes = await request.body()
        body_str = body_bytes.decode('utf-8') if body_bytes else ""
        
        # Try to parse as JSON
        body = None
        if body_str:
            try:
                # Try parsing as JSON
                body = json.loads(body_str)
            except json.JSONDecodeError:
                # If it's not valid JSON, it might be sent as a string representation
                # Try to evaluate it (but be safe about it)
                try:
                    # Remove any surrounding quotes if present
                    cleaned = body_str.strip()
                    if cleaned.startswith('"') and cleaned.endswith('"'):
                        cleaned = cleaned[1:-1]
                    # Try parsing again
                    body = json.loads(cleaned)
                except:
                    # Last resort: try to parse as form data
                    try:
                        # Reconstruct request for form parsing
                        from fastapi import Form
                        # This won't work since body is already consumed, so we'll handle it differently
                        pass
                    except:
                        pass
        
        # If we still don't have a body, try to get it from request.json() (might work if FastAPI cached it)
        if body is None:
            try:
                body = await request.json()
            except:
                pass
        
        # If still no body, return helpful error
        if body is None or not isinstance(body, dict):
            raise HTTPException(
                status_code=422,
                detail=f"Could not parse request body. Received: {body_str[:200] if body_str else 'empty'}. Expected JSON object with user_id and image_url."
            )
        
        # Extract user_id and image_url
        final_user_id = body.get("user_id")
        final_image_url = body.get("image_url")
        
        # Check if fields are missing or empty
        if not final_user_id:
            raise HTTPException(
                status_code=422, 
                detail="user_id is required in the request body"
            )
        
        if not final_image_url:
            raise HTTPException(
                status_code=422, 
                detail="image_url is required in the request body and cannot be empty"
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

