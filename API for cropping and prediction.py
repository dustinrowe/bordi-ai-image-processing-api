import os
import io
import json
import requests
import base64
from datetime import datetime
from PIL import Image
import numpy as np
from supabase import create_client, Client

def get_max_side_length(pts):
    """Get width and height for the quadrilateral as max of opposite sides"""
    pts = np.array(pts)
    w1 = np.linalg.norm(pts[0] - pts[1])
    w2 = np.linalg.norm(pts[2] - pts[3])
    w = int(round(max(w1, w2)))
    h1 = np.linalg.norm(pts[0] - pts[3])
    h2 = np.linalg.norm(pts[1] - pts[2])
    h = int(round(max(h1, h2)))
    return w, h

def get_perspective_transform(src, dst):
    """Emulate cv2.getPerspectiveTransform using numpy"""
    assert src.shape == (4, 2)
    assert dst.shape == (4, 2)
    A = []
    for i in range(4):
        x, y = src[i][0], src[i][1]
        u, v = dst[i][0], dst[i][1]
        A.extend([
            [-x, -y, -1,   0,  0,  0, x*u, y*u, u],
            [ 0,  0,  0,  -x, -y, -1, x*v, y*v, v]
        ])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    M = L.reshape(3, 3)
    return M

def warp_perspective(img, mat, dsize):
    """Emulate cv2.warpPerspective via PIL and numpy"""
    inv_mat = np.linalg.inv(mat)
    h_out, w_out = dsize[1], dsize[0]
    # grid of destination points
    xx, yy = np.meshgrid(np.arange(w_out), np.arange(h_out))
    homog = np.stack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()], axis=1).T
    src_coords = inv_mat @ homog
    src_coords /= src_coords[2]
    x_src = src_coords[0].reshape((h_out, w_out))
    y_src = src_coords[1].reshape((h_out, w_out))
    # Clipping
    x_src_clip = np.clip(x_src, 0, img.shape[1] - 1)
    y_src_clip = np.clip(y_src, 0, img.shape[0] - 1)
    xq = np.round(x_src_clip).astype(np.int32)
    yq = np.round(y_src_clip).astype(np.int32)
    # handle grayscale/RGB
    if len(img.shape) == 3:
        sampled = img[yq, xq, :]
    else:
        sampled = img[yq, xq]
    return sampled

def crop_and_warp(image, src_points, output_size):
    """Given 4 src_points in (x, y), warp to rectangle of output_size"""
    src = np.array(src_points, dtype=np.float32)
    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype=np.float32)
    mat = get_perspective_transform(src, dst)
    img_np = np.array(image)
    warped = warp_perspective(img_np, mat, output_size)
    return Image.fromarray(warped)

def process_habit_image(user_id: str, image_url: str):
    """
    Process a habit tracking image by cropping cells and classifying them with Roboflow.
    
    Args:
        user_id: The user ID to look up crop values in image_crop_values table
        image_url: The URL of the image to process
    
    Returns:
        dict: JSON object with classification results, statistics, and timestamp
    """
    # Initialize Supabase client
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url:
        raise ValueError("SUPABASE_URL environment variable is required")
    if not supabase_key:
        raise ValueError("SUPABASE_KEY environment variable is required")
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Query image_crop_values table using user_id, get the most recently created one
    # Try common timestamp column names (created_at is most common in Supabase)
    try:
        response = supabase.table('image_crop_values').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(1).execute()
    except Exception:
        # If created_at doesn't exist, try with id (assuming higher id = more recent)
        try:
            response = supabase.table('image_crop_values').select('*').eq('user_id', user_id).order('id', desc=True).limit(1).execute()
        except Exception:
            # Fallback: just get any record matching user_id
            response = supabase.table('image_crop_values').select('*').eq('user_id', user_id).limit(1).execute()
    
    if not response.data or len(response.data) == 0:
        raise ValueError(f"No crop values found for user_id: {user_id}")
    
    # Get the crop values record
    crop_record = response.data[0]
    
    # Extract crop data - the crop values are stored as an array of cell objects
    # Check common column names that might contain the array
    cells = None
    for possible_key in ['crop_values', 'values', 'cells', 'data', 'crop_data']:
        if possible_key in crop_record and isinstance(crop_record[possible_key], list):
            cells = crop_record[possible_key]
            break
    
    # If not found in a column, check if the record itself is the array structure
    if cells is None:
        # Check if crop_record is already an array (unlikely but possible)
        if isinstance(crop_record, list):
            cells = crop_record
        else:
            # Try to find any array field in the record
            for key, value in crop_record.items():
                if isinstance(value, list) and len(value) > 0:
                    # Check if it looks like a cells array (has objects with row/col/points)
                    if isinstance(value[0], dict) and 'row' in value[0] and 'col' in value[0]:
                        cells = value
                        break
    
    if not cells:
        raise ValueError(f"No crop values array found in image_crop_values record for user_id: {user_id}. Available keys: {list(crop_record.keys())}")
    
    image_dimensions = crop_record.get('image_dimensions', [2500, 2500])
    
    # Validate required fields
    if not isinstance(cells, list) or len(cells) == 0:
        raise ValueError("crop values must be a non-empty array")
    if not image_url:
        raise ValueError("image_url parameter is required")
    
    # Use the passed image_url instead of getting it from the database
    img_url = image_url

    # Download image into a PIL Image
    resp = requests.get(img_url)
    img = Image.open(io.BytesIO(resp.content)).convert('RGB')

    # Roboflow API endpoint - get from environment variable
    roboflow_url = os.getenv('ROBOFLOW_API_URL')
    if not roboflow_url:
        raise ValueError("ROBOFLOW_API_URL environment variable is required")

    # Create directory for cropped images in Downloads folder
    downloads_path = os.path.expanduser("~/Downloads")
    timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    cropped_images_dir = os.path.join(downloads_path, f"cropped_cells_{timestamp_str}")
    os.makedirs(cropped_images_dir, exist_ok=True)

    # Collect all results
    results = []
    all_confidences = []
    total_true = 0
    total_false = 0

    for cell in cells:
        pts = cell["points"]
        w, h = get_max_side_length(pts)
        cropped = crop_and_warp(img, pts, (w, h))
        
        # Save cropped image to file
        cell_id = f"r{cell['row']}c{cell['col']}"
        cropped_filename = f"{cell_id}.png"
        cropped_filepath = os.path.join(cropped_images_dir, cropped_filename)
        cropped.save(cropped_filepath)
        
        # Convert cropped image to base64
        buffer = io.BytesIO()
        cropped.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Send to Roboflow API (base64 data as raw body, like curl -d @-)
        response = requests.post(
            roboflow_url,
            data=img_base64,
            headers={'Content-Type': 'text/plain'}
        )
        
        # Parse response (cell_id already defined above)
        predictions = []
        
        if response.status_code == 200:
            try:
                api_result = response.json()
                # Handle different possible response formats
                if isinstance(api_result, list):
                    predictions = api_result
                elif isinstance(api_result, dict):
                    if 'predictions' in api_result:
                        predictions = api_result['predictions']
                    elif 'class' in api_result or 'confidence' in api_result:
                        # Single prediction object
                        predictions = [api_result]
                    else:
                        predictions = [api_result]
                else:
                    predictions = []
                
                # Extract confidence and class for statistics (one prediction per cell)
                if predictions and isinstance(predictions[0], dict):
                    pred = predictions[0]
                    confidence = pred.get('confidence', 0.0)
                    all_confidences.append(confidence)
                    pred_class = str(pred.get('class', '')).lower()
                    if pred_class == 'true':
                        total_true += 1
                    elif pred_class == 'false':
                        total_false += 1
                    
                    # Upload low-confidence crops to Roboflow for retraining
                    if confidence < 0.85:
                        try:
                            roboflow_api_key = os.getenv('ROBOFLOW_UPLOAD_API_KEY')
                            roboflow_project = os.getenv('ROBOFLOW_PROJECT')
                            
                            if roboflow_api_key and roboflow_project:
                                # Read image as base64
                                with open(cropped_filepath, 'rb') as img_file:
                                    img_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                                
                                upload_url = (
                                    f"https://api.roboflow.com/dataset/{roboflow_project}/upload"
                                    f"?api_key={roboflow_api_key}"
                                    f"&name={cell_id}.png"
                                    f"&split=train"
                                    f"&tag=low_confidence"
                                    f"&tag=conf_{int(confidence*100)}"
                                )
                                upload_resp = requests.post(
                                    upload_url,
                                    data=img_b64,
                                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                                )
                                if upload_resp.status_code == 200:
                                    print(f"Uploaded {cell_id} to Roboflow (conf: {confidence:.2f})")
                                else:
                                    print(f"Roboflow upload failed for {cell_id}: {upload_resp.status_code} {upload_resp.text}")
                        except Exception as upload_error:
                            # Don't fail the entire request if upload fails
                            print(f"Failed to upload {cell_id} to Roboflow: {upload_error}")
            except Exception as e:
                # If parsing fails, continue with empty predictions
                pass
        
        results.append({
            "id": cell_id,
            "predictions": predictions
        })
    
    # Calculate average confidence
    average_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    low_confidence_threshold = 0.85
    low_confidence_count = sum(1 for confidence in all_confidences if confidence < low_confidence_threshold)
    
    # Generate timestamp
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Build output
    output = {
        "image_url": img_url,
        "timestamp": timestamp,
        "average_confidence": round(average_confidence, 4),
        "low_confidence_count": low_confidence_count,
        "total_true": total_true,
        "total_false": total_false,
        "results": results,
        "cropped_images_directory": cropped_images_dir
    }
    
    return output


def main():
    """
    Example usage of the process_habit_image function.
    For API usage, call process_habit_image(user_id, image_url) directly.
    """
    import sys
    
    # ============================================
    # EDIT THESE VALUES TO CHANGE INPUT:
    # ============================================
    DEFAULT_USER_ID = "f9370f47-ae09-4b29-a42b-44aef42d5389"  # Edit this with your user_id
    DEFAULT_IMAGE_URL = "https://res.cloudinary.com/db6fegsqa/image/upload/v1763427064/wbzlfsefitkddzqzv5y2.jpg"  # Edit this with your image URL
    # ============================================
    
    # Example usage - can be called from command line or API
    if len(sys.argv) >= 3:
        user_id = sys.argv[1]
        image_url = sys.argv[2]
    else:
        # Use environment variables or default values
        user_id = os.getenv('TEST_USER_ID', DEFAULT_USER_ID)
        image_url = os.getenv('TEST_IMAGE_URL', DEFAULT_IMAGE_URL)
        
        if not user_id or not image_url:
            print("Usage: python script.py <user_id> <image_url>")
            print("Or set TEST_USER_ID and TEST_IMAGE_URL environment variables")
            print("Or edit DEFAULT_USER_ID and DEFAULT_IMAGE_URL in the main() function")
            return
    
    try:
        result = process_habit_image(user_id, image_url)
        
        # Optionally save to file for debugging
        downloads_path = os.path.expanduser("~/Downloads")
        filename = f"roboflow_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        output_file = os.path.join(downloads_path, filename)
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        if 'cropped_images_directory' in result:
            print(f"Cropped images saved to: {result['cropped_images_directory']}")
        print("\n" + json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
