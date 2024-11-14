from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# Model details
PREDICTION_ENDPOINT = "https://detectfruits-prediction.cognitiveservices.azure.com"
PREDICTION_KEY = "9RJamSwtnHGpA63YOX3Z9IYmPCfE65llyBkVPKNsv3cqObSEEbMOJQQJ99AKACYeBjFXJ3w3AAAIACOGZtEK"
PROJECT_ID = "b240521c-4fa7-45ac-9210-1729cca66ec9"
MODEL_NAME = "Iteration1"

# Load and authenticate with the prediction client
credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
prediction_client = CustomVisionPredictionClient(endpoint=PREDICTION_ENDPOINT, credentials=credentials)

def detect_objects(image_path):
    # Open image file
    with open(image_path, "rb") as image_data:
        # Perform object detection
        results = prediction_client.detect_image(PROJECT_ID, MODEL_NAME, image_data)
    
    # Load image for displaying results
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    w, h = image.size

    # Set font for the labels (increase size)
    try:
        font = ImageFont.truetype("arial.ttf", 20)  # Adjust font size as needed
    except IOError:
        font = ImageFont.load_default()  # Default font if custom font is unavailable

    # Process each prediction
    for prediction in results.predictions:
        # Only show predictions with a high probability
        if prediction.probability > 0.5:
            # Get bounding box coordinates (convert from ratio to actual values)
            left = int(prediction.bounding_box.left * w)
            top = int(prediction.bounding_box.top * h)
            width = int(prediction.bounding_box.width * w)
            height = int(prediction.bounding_box.height * h)
            
            # Draw bounding box and label
            box_color = "red"
            draw.rectangle([(left, top), (left + width, top + height)], outline=box_color, width=3)
            label = f"{prediction.tag_name} ({prediction.probability * 100:.1f}%)"
            text_size = draw.textsize(label, font=font)
            draw.rectangle([left, top - text_size[1], left + text_size[0], top], fill=box_color)
            draw.text((left, top - text_size[1]), label, fill="white", font=font)
    
    # Save the output image with bounding boxes
    output_file = "output_with_detections.jpg"
    image.save(output_file)
    print(f"Results saved in {output_file}")

    # Display the saved image with matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(np.array(image))
    plt.axis("off")
    plt.show()

# Run the detection on the specified image
detect_objects("produce.jpg")
