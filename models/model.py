import torch
import torch.nn as nn
from torchvision import models
import cv2

class XRayAnalysisModel(nn.Module):
    def __init__(self, num_classes=2):
        super(XRayAnalysisModel, self).__init__()
        # Load the pretrained ResNet18 model
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        # Modify the final layer to match the number of classes
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Add dropout to prevent overfitting
            nn.Linear(self.model.fc.in_features, num_classes)  # Adjust output layer
        )

    def forward(self, x):
        return self.model(x)


def load_and_preprocess_image(image_path):
    # Load the image in BGR format and convert to RGB
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded. Please check the file path or format.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # Apply histogram equalization to enhance contrast
    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    channels = list(cv2.split(image_yuv))  # Convert tuple to list for mutability
    channels[0] = cv2.equalizeHist(channels[0])  # Equalize the Y channel
    image_yuv = cv2.merge(channels)
    image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

    # Normalize the image to [0, 1] range and rearrange dimensions to (C, H, W)
    image = image / 255.0
    image = torch.FloatTensor(image).permute(2, 0, 1)  # Change shape to (C, H, W)

    return image
