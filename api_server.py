"""
FastAPI server for habit image processing API.
Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
Or: python api_server.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import sys
import os
import requests
import cv2
import numpy as np

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

# Import image quality checker
quality_spec = importlib.util.spec_from_file_location(
    "image_quality_check",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_quality_check.py")
)
quality_checker = importlib.util.module_from_spec(quality_spec)
quality_spec.loader.exec_module(quality_checker)
readability_prefilter = quality_checker.readability_prefilter

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


class QualityCheckRequest(BaseModel):
    image_urls: List[HttpUrl]

    class Config:
        json_schema_extra = {
            "example": {
                "image_urls": [
                    "https://res.cloudinary.com/db6fegsqa/image/upload/v1770151294/snnpozu2gyfp4kc4mw0a.jpg",
                    "https://res.cloudinary.com/db6fegsqa/image/upload/v1770167953/axjknikvdjkvjk6qcfop.jpg",
                    "https://res.cloudinary.com/db6fegsqa/image/upload/v1770163705/pmmmrk4o8qvjnqb63tda.jpg",
                    "https://res.cloudinary.com/db6fegsqa/image/upload/v1770139038/f1ruw5okmqmocbux7wok.jpg"
                ]
            }
        }


class QualityCheckResult(BaseModel):
    image_url: str
    ok: bool
    metrics: dict
    error: Optional[str] = None


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
async def process_image(request_body: ProcessImageRequest):
    """
    Process a habit tracking image by cropping cells and classifying them.
    
    Accepts JSON body:
    - **user_id**: The user ID to look up crop values in image_crop_values table
    - **image_url**: The URL of the image to process
    
    Returns classification results, statistics, and timestamp.
    """
    try:
        result = process_habit_image(request_body.user_id, str(request_body.image_url))
        return result
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


@app.post("/quality-check", response_model=List[QualityCheckResult])
async def quality_check(request_body: QualityCheckRequest):
    """
    Evaluate image quality for multiple images.
    Returns pass/fail plus metrics for each image.
    """
    results: List[QualityCheckResult] = []
    for image_url in request_body.image_urls:
        try:
            response = requests.get(str(image_url), timeout=15)
            response.raise_for_status()
            data = np.frombuffer(response.content, dtype=np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img_bgr is None:
                results.append(
                    QualityCheckResult(
                        image_url=str(image_url),
                        ok=False,
                        metrics={},
                        error="unreadable"
                    )
                )
                continue

            ok, metrics = readability_prefilter(img_bgr)
            results.append(
                QualityCheckResult(
                    image_url=str(image_url),
                    ok=ok,
                    metrics=metrics,
                )
            )
        except requests.RequestException:
            results.append(
                QualityCheckResult(
                    image_url=str(image_url),
                    ok=False,
                    metrics={},
                    error="download_failed"
                )
            )
        except Exception as e:
            results.append(
                QualityCheckResult(
                    image_url=str(image_url),
                    ok=False,
                    metrics={},
                    error=str(e)
                )
            )

    return results


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting API server on http://{host}:{port}")
    print(f"API documentation available at http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)

