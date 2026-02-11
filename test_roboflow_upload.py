"""
Test script to verify Roboflow upload works for low-confidence crops.
"""
import os
import sys

# Set environment variables for testing
os.environ['SUPABASE_URL'] = 'https://cixjwghnddberuigvls.supabase.co'
os.environ['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNpeGp3Z2huZGRiZXJ1aWd2bHMiLCJyb2xlIjoiYW5vbiIsImlhdCI6MTczMzQ3MDA4MCwiZXhwIjoyMDQ5MDQ2MDgwfQ.yJhbGciOiJzdXBhYmFzZSIsInJlZiI6ImNpeGp3Z2huZGRiZXJ1aWd2bHMiLCJyb2xlIjoiYW5vbiIsImlhdCI6MTczMzQ3MDA4MCwiZXhwIjoyMDQ5MDQ2MDgwfQ'
os.environ['ROBOFLOW_API_URL'] = 'https://classify.roboflow.com/ischecked-onblp/5?api_key=n6ff5cSCNfbFLjLKhGyF'
os.environ['ROBOFLOW_UPLOAD_API_KEY'] = 'n6ff5cSCNfbFLjLKhGyF'
os.environ['ROBOFLOW_WORKSPACE'] = 'mosaiclyai'
os.environ['ROBOFLOW_PROJECT'] = 'ischecked-onblp'

# Import after setting env vars
import importlib.util
spec = importlib.util.spec_from_file_location(
    "habit_processor",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "API for cropping and prediction.py")
)
habit_processor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(habit_processor)

# Test with an existing user and image
TEST_USER_ID = "da481463-eaa9-444c-a0d0-fc6a434a8317"
TEST_IMAGE_URL = "https://res.cloudinary.com/db6fegsqa/image/upload/v1770678826/rxdzl1spjadxllqv4squ.jpg"

print("Starting Roboflow upload test...")
print(f"User ID: {TEST_USER_ID}")
print(f"Image URL: {TEST_IMAGE_URL}")
print(f"Roboflow Project: {os.environ['ROBOFLOW_WORKSPACE']}/{os.environ['ROBOFLOW_PROJECT']}")
print("\nProcessing image...")

try:
    result = habit_processor.process_habit_image(TEST_USER_ID, TEST_IMAGE_URL)
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Average Confidence: {result['average_confidence']}")
    print(f"Low Confidence Count: {result['low_confidence_count']}")
    print(f"Total True: {result['total_true']}")
    print(f"Total False: {result['total_false']}")
    print(f"\nProcessed {len(result['results'])} crops")
    
    # Count how many should have been uploaded
    low_conf_crops = [r for r in result['results'] if r['predictions'] and r['predictions'][0].get('confidence', 1.0) < 0.85]
    print(f"\nCrops with confidence < 0.85: {len(low_conf_crops)}")
    
    if low_conf_crops:
        print("\nLow confidence crops that should be uploaded to Roboflow:")
        for crop in low_conf_crops:
            pred = crop['predictions'][0]
            conf = pred.get('confidence', 0)
            cls = pred.get('class', 'unknown')
            print(f"  - {crop['id']}: {cls} ({conf:.2%})")
        
        print("\n✅ Check your Roboflow project to verify uploads:")
        print(f"   https://app.roboflow.com/{os.environ['ROBOFLOW_WORKSPACE']}/{os.environ['ROBOFLOW_PROJECT']}/annotate")
    else:
        print("\n⚠️  No low-confidence crops found in this image. Try another image with lower confidence scores.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
