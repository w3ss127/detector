import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Assuming ResNetViTHybrid is defined as in train_model.py
from train_model import ResNetViTHybrid, NUM_CLASSES, RESNET_VARIANT

# Define test transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ResNetViTHybrid(NUM_CLASSES, resnet_variant=RESNET_VARIANT)
checkpoint = torch.load("checkpoints/best_model.pt", map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# Function to preprocess and test a single image tensor
def test_single_image(image_tensor):
    # Input: uint8 tensor, shape [3, 256, 256] or [256, 256, 3]
    image_tensor = torch.tensor(image_tensor, dtype=torch.uint8)
    if image_tensor.shape[-1] == 3:  # HWC to CHW
        image_tensor = image_tensor.permute(2, 0, 1)
    image = image_tensor.float() / 255.0  # Convert to float32, scale to [0, 1]
    image = test_transform(image)  # Apply resize and normalization
    image = image.unsqueeze(0).to(DEVICE)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
    return prediction, probabilities.cpu().numpy()[0]

# Example usage
# Assuming image_tensor is a uint8 numpy array or tensor, shape [256, 256, 3]
image_tensor = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)  # Example image
prediction, probs = test_single_image(image_tensor)
class_names = ["real", "synthetic", "semi-synthetic"]
print(f"Prediction: {class_names[prediction]}")
print(f"Probabilities: {', '.join([f'{name}: {prob:.4f}' for name, prob in zip(class_names, probs)])}")