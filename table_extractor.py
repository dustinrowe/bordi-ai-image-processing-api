from paddleocr import TableRecognitionPipelineV2
import os
import json
import numpy as np
from datetime import datetime

def extract_table(image_path, save_results=True):
    # Initialize the table recognition pipeline
    pipeline = TableRecognitionPipelineV2()
    
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    try:
        # Process the image using the predict method
        result = pipeline.predict(image_path)
        
        # Save results if requested
        if save_results and result:
            save_results_to_file(result, image_path)
        
        return result
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def convert_numpy_to_json_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json_serializable(item) for item in obj]
    else:
        return obj

def save_results_to_file(result, image_path):
    # Create results directory if it doesn't exist
    results_dir = "table_extraction_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"table_result_{timestamp}"
    
    # Convert result to JSON-serializable format
    json_serializable_result = convert_numpy_to_json_serializable(result)
    
    # Save as JSON
    json_path = os.path.join(results_dir, f"{base_filename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_serializable_result, f, indent=2, ensure_ascii=False)
    
    # Save as text
    txt_path = os.path.join(results_dir, f"{base_filename}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Table Extraction Results\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write("-" * 50 + "\n")
        f.write(str(result))
    
    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Text: {txt_path}")

if __name__ == "__main__":
    image_path = '/Users/dflow/Downloads/7a7e641c-306e-4fd1-af28-d26b812f0bee250.jpg'
    result = extract_table(image_path)
    print("Table Recognition Results:")
    print(result)