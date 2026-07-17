# test_model.py - Works with JPG, PNG, BMP, etc.
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from improved_pest_detection import PestDetectionSystem
import cv2
import glob

print("🔍 Testing 99.10% Plant Disease Model...")
print("="*60)

# Check files
required = ['tinylitenet_best.keras', 'class_names.json', 'improved_pest_detection.py']
for file in required:
    if os.path.exists(file):
        print(f"✅ Found: {file}")
    else:
        print(f"❌ Missing: {file}")
        exit(1)

# Load class names
with open('class_names.json', 'r') as f:
    class_names = json.load(f)
print(f"\n📋 Loaded {len(class_names)} classes")

# Load model
print("\n🤖 Loading model...")
system = PestDetectionSystem(
    model_path='tinylitenet_best.keras',
    class_names=class_names
)
print("✅ Model loaded!")

# OPTION 1: Test a specific image
test_image = 'test_leaf.jpeg'  # Change this to your image name
# Or use: test_image = 'test_plant.png'
# Or use: test_image = 'my_photo.bmp'

if os.path.exists(test_image):
    print(f"\n🔍 Testing specific image: {test_image}")
    result = system.predict_from_image_path(test_image, show_steps=True)
    
    # Print results
    print("\n" + "="*60)
    print(f"📁 File: {test_image}")
    print(f"🌿 Disease: {result['predicted_class']}")
    print(f"📊 Confidence: {result['confidence']:.2f}%")
    print("="*60)
    
else:
    print(f"\n⚠️  '{test_image}' not found")

# OPTION 2: Find and test ALL images in folder
print("\n🔍 Looking for any images in current folder...")
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
image_files = []

for ext in image_extensions:
    image_files.extend(glob.glob(ext))
    image_files.extend(glob.glob(ext.upper()))  # Also check uppercase

if image_files:
    print(f"\n📸 Found {len(image_files)} images:")
    for i, img_file in enumerate(image_files[:3]):  # Show first 3
        print(f"   {i+1}. {img_file}")
    
    # Test the first image
    print(f"\n🔍 Testing first image: {image_files[0]}")
    result = system.predict_from_image_path(image_files[0], show_steps=True)
    
    print("\n" + "="*60)
    print(f"📁 File: {image_files[0]}")
    print(f"🌿 Disease: {result['predicted_class']}")
    print(f"📊 Confidence: {result['confidence']:.2f}%")
    print("="*60)
    
    # Show prediction chart
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Top predictions
    plt.subplot(1, 2, 1)
    probs = result['all_probs']
    top10_idx = np.argsort(probs)[-10:][::-1]
    top10_names = [class_names[i][:20] + '...' if len(class_names[i]) > 20 else class_names[i] for i in top10_idx]
    top10_probs = probs[top10_idx] * 100
    
    bars = plt.barh(range(len(top10_probs)), top10_probs)
    plt.yticks(range(len(top10_probs)), top10_names)
    plt.xlabel('Confidence (%)')
    plt.title('Top 10 Predictions')
    bars[0].set_color('green')
    
    # Plot 2: Original image
    plt.subplot(1, 2, 2)
    img = cv2.imread(image_files[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Pred: {result['predicted_class']}\nConf: {result['confidence']:.1f}%")
    
    plt.tight_layout()
    plt.show()
    
else:
    print("\n📸 No images found in folder!")
    print("   Please add any image (JPG, PNG, BMP, etc.) to test")

# OPTION 3: Interactive - Type image name
print("\n" + "="*60)
print("📝 OPTIONAL: Type an image filename to test")
print("   (or press Enter to skip)")
custom_image = input("Image name: ").strip()

if custom_image and os.path.exists(custom_image):
    print(f"\n🔍 Testing: {custom_image}")
    result = system.predict_from_image_path(custom_image)
    print(f"\n🌿 Disease: {result['predicted_class']}")
    print(f"📊 Confidence: {result['confidence']:.2f}%")