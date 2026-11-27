import torch
from torchvision import transforms
from PIL import Image
from predictor import load_model


# Cassava disease classes
classes = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mite (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy"
]


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load trained model
model = load_model()

print("Model loaded successfully!")


# Load test image
image_path = "test_images/sample.jpg"

image = Image.open(image_path).convert("RGB")


# Apply preprocessing
image = transform(image)


# Add batch dimension
image = image.unsqueeze(0)


# Prediction
model.eval()

with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(output, dim=1)

    confidence, prediction = torch.max(probabilities, dim=1)


# Display result
predicted_class = prediction.item()
confidence_score = confidence.item() * 100


print("--------------------------------")
print("Cassava Disease Prediction")
print("--------------------------------")

print("Disease:", classes[predicted_class])
print(f"Confidence: {confidence_score:.2f}%")