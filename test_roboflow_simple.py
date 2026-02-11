"""
Simple test to verify Roboflow upload works.
Creates a dummy image and uploads it directly.
"""
import os
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow

# Set Roboflow credentials
ROBOFLOW_API_KEY = 'n6ff5cSCNfbFLjLKhGyF'
ROBOFLOW_WORKSPACE = 'mosaiclyai'
ROBOFLOW_PROJECT = 'ischecked-onblp'

print("Creating test image...")
# Create a simple test image
img = Image.new('RGB', (200, 200), color='white')
draw = ImageDraw.Draw(img)
draw.rectangle([50, 50, 150, 150], fill='gray', outline='black', width=3)
draw.text((70, 90), "TEST", fill='black')

# Save to temp file
test_image_path = "/tmp/roboflow_test_crop.png"
img.save(test_image_path)
print(f"Saved test image to: {test_image_path}")

print(f"\nConnecting to Roboflow...")
print(f"Workspace: {ROBOFLOW_WORKSPACE}")
print(f"Project: {ROBOFLOW_PROJECT}")

try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    
    print("\nUploading test image...")
    result = project.upload(
        image_path=test_image_path,
        annotation_path=None,
        split="train",
        tag_names=["test_upload", "low_confidence", "conf_75"]
    )
    
    print("\n✅ SUCCESS! Image uploaded to Roboflow")
    print(f"\nCheck your Roboflow project:")
    print(f"https://app.roboflow.com/{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}/annotate")
    print(f"\nLook for the image tagged with: test_upload, low_confidence, conf_75")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
