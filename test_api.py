"""
Simple test script for the Habit Image Processing API.
Run this after starting the API server to test the endpoints.
"""

import requests
import json
import sys

# Default API URL - change if your server is running on a different host/port
API_URL = "http://localhost:8000"

def test_health():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200

def test_process_image_post():
    """Test the POST /process-image endpoint"""
    print("Testing POST /process-image endpoint...")
    
    # Example data - replace with your actual values
    data = {
        "user_id": "f9370f47-ae09-4b29-a42b-44aef42d5389",
        "image_url": "https://res.cloudinary.com/db6fegsqa/image/upload/v1763427064/wbzlfsefitkddzqzv5y2.jpg"
    }
    
    try:
        response = requests.post(
            f"{API_URL}/process-image",
            json=data,
            timeout=60  # Processing may take a while
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Processed {len(result.get('results', []))} cells")
            print(f"Total True: {result.get('total_true', 0)}")
            print(f"Total False: {result.get('total_false', 0)}")
            print(f"Average Confidence: {result.get('average_confidence', 0)}")
            print(f"\nFull response saved to test_response.json")
            with open("test_response.json", "w") as f:
                json.dump(result, f, indent=2)
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False

def test_process_image_get():
    """Test the GET /process-image endpoint"""
    print("Testing GET /process-image endpoint...")
    
    params = {
        "user_id": "f9370f47-ae09-4b29-a42b-44aef42d5389",
        "image_url": "https://res.cloudinary.com/db6fegsqa/image/upload/v1763427064/wbzlfsefitkddzqzv5y2.jpg"
    }
    
    try:
        response = requests.get(
            f"{API_URL}/process-image",
            params=params,
            timeout=60
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Processed {len(result.get('results', []))} cells")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        API_URL = sys.argv[1]
    
    print(f"Testing API at: {API_URL}\n")
    print("=" * 50)
    
    # Test health endpoint
    if not test_health():
        print("Health check failed. Is the server running?")
        sys.exit(1)
    
    # Test POST endpoint
    print("=" * 50)
    test_process_image_post()
    
    # Test GET endpoint (optional - comment out if you don't want to test it)
    # print("=" * 50)
    # test_process_image_get()
    
    print("=" * 50)
    print("Testing complete!")

