import tensorflow as tf
import numpy as np
from PIL import Image as PILImage
import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import ImageForm
import base64
from io import BytesIO

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'model.h5')
model = tf.keras.models.load_model(MODEL_PATH)

# Custom image labels and filenames (use your own image names here)
labels = ['Airplane', 'Bird', 'Car', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
image_folder = os.path.join(os.path.dirname(__file__), 'static')  # Path to static folder

def classify_image(image_path):
    try:
        # Preprocess the uploaded image
        img = PILImage.open(image_path).convert('RGB').resize((32, 32))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict the class
        prediction = model.predict(img)
        return labels[np.argmax(prediction)]
    except Exception as e:
        return "Error: Unable to process the image."

def get_sample_images():
    # Fetch images from the custom folder (using the provided image paths)
    sample_images_info = []
    for label in labels:
        image_path = os.path.join(image_folder, f"{label}.jpg")  # Assuming images are named like 'Airplane.jpg'
        if os.path.exists(image_path):
            # Open the image
            img = PILImage.open(image_path)

            # Save image to a BytesIO object for base64 encoding
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')  # Convert to base64 string
            
            sample_images_info.append({'image': img_str, 'label': label})

    return sample_images_info

def index(request):
    # Get custom sample images for display
    sample_images_info = get_sample_images()

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']

        # Validate file type
        if not image.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render(request, 'index.html', {'error': 'Invalid file format. Please upload a PNG or JPG image.'})

        # Save the image
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        uploaded_image_url = fs.url(filename)

        # Classify the image
        result = classify_image(fs.path(filename))

        # Delete the file after processing
        fs.delete(filename)

        return render(request, 'index.html', {'uploaded_image_url': uploaded_image_url, 'result': result, 'sample_images_info': sample_images_info})

    return render(request, 'index.html', {'sample_images_info': sample_images_info})
