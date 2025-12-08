# Habit Image Processing API

A FastAPI-based REST API for processing habit tracking images with automatic cropping and classification.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the API Server

**Option A: Using Python directly**
```bash
python api_server.py
```

**Option B: Using uvicorn directly**
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: `http://localhost:8000`
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### 3. Test the API

**Using curl (POST request):**
```bash
curl -X POST "http://localhost:8000/process-image" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "f9370f47-ae09-4b29-a42b-44aef42d5389",
    "image_url": "https://res.cloudinary.com/db6fegsqa/image/upload/v1763427064/wbzlfsefitkddzqzv5y2.jpg"
  }'
```

**Using curl (GET request):**
```bash
curl "http://localhost:8000/process-image?user_id=f9370f47-ae09-4b29-a42b-44aef42d5389&image_url=https://res.cloudinary.com/db6fegsqa/image/upload/v1763427064/wbzlfsefitkddzqzv5y2.jpg"
```

**Using Python requests:**
```python
import requests

response = requests.post(
    "http://localhost:8000/process-image",
    json={
        "user_id": "f9370f47-ae09-4b29-a42b-44aef42d5389",
        "image_url": "https://res.cloudinary.com/db6fegsqa/image/upload/v1763427064/wbzlfsefitkddzqzv5y2.jpg"
    }
)

print(response.json())
```

## API Endpoints

### POST `/process-image`
Process a habit tracking image.

**Request Body:**
```json
{
  "user_id": "string",
  "image_url": "https://example.com/image.jpg"
}
```

**Response:**
```json
{
  "image_url": "https://example.com/image.jpg",
  "timestamp": "2025-01-30T12:00:00Z",
  "average_confidence": 0.95,
  "total_true": 5,
  "total_false": 2,
  "results": [
    {
      "id": "r0c0",
      "predictions": [...]
    }
  ],
  "cropped_images_directory": "/path/to/cropped/images"
}
```

### GET `/process-image`
Alternative GET endpoint with query parameters:
- `user_id` (required)
- `image_url` (required)

### GET `/health`
Health check endpoint.

### GET `/`
Root endpoint with API information.

## Environment Variables

You can set these environment variables to configure the API:

- `SUPABASE_URL`: Supabase project URL (defaults to existing value)
- `SUPABASE_KEY`: Supabase service role key (defaults to existing value)
- `PORT`: Server port (defaults to 8000)
- `HOST`: Server host (defaults to 0.0.0.0)

## Deployment Options

### Option 1: Local Development
Just run `python api_server.py` or `uvicorn api_server:app --host 0.0.0.0 --port 8000`

### Option 2: Deploy to Cloud Platforms

**Railway:**
1. Create a `Procfile` with: `web: uvicorn api_server:app --host 0.0.0.0 --port $PORT`
2. Push to GitHub and connect to Railway
3. Set environment variables in Railway dashboard

**Render:**
1. Create a `render.yaml` or use the web interface
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`

**Heroku:**
1. Create a `Procfile` with: `web: uvicorn api_server:app --host 0.0.0.0 --port $PORT`
2. Deploy using Heroku CLI or GitHub integration

**AWS/GCP/Azure:**
- Use container services (ECS, Cloud Run, Container Instances)
- Or use serverless options (Lambda with API Gateway, Cloud Functions, Azure Functions)

### Option 3: Docker (Recommended for Production)

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Then build and run:
```bash
docker build -t habit-api .
docker run -p 8000:8000 habit-api
```

## Security Notes

⚠️ **Important:** The current CORS configuration allows all origins (`allow_origins=["*"]`). For production:

1. **Restrict CORS origins:**
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://yourdomain.com", "https://app.yourdomain.com"],
       allow_credentials=True,
       allow_methods=["POST", "GET"],
       allow_headers=["*"],
   )
   ```

2. **Add authentication/API keys** if needed:
   - Use FastAPI's dependency system for API key validation
   - Or add OAuth2/JWT authentication

3. **Rate limiting:** Consider adding rate limiting to prevent abuse

4. **Environment variables:** Never commit sensitive keys to version control

## Troubleshooting

- **Port already in use:** Change the port with `PORT=8080 python api_server.py`
- **Import errors:** Make sure all dependencies are installed with `pip install -r requirements.txt`
- **CORS errors:** Check that the CORS middleware is properly configured
- **Supabase connection issues:** Verify your `SUPABASE_URL` and `SUPABASE_KEY` environment variables

