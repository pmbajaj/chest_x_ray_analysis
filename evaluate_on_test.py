import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from models.model import XRayAnalysisModel
from train import XRayDataset
import time


# Function to evaluate the model on the test set
def evaluate_on_test(model_path, test_data_dir, batch_size=32, device=None):
    # Set device to GPU if available, otherwise CPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the same data transformation as used in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the test dataset
    test_dataset = XRayDataset(root_dir=test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model and load the best saved model checkpoint
    model = XRayAnalysisModel(num_classes=len(test_dataset.class_names)).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Evaluate the model on the test set
    y_pred, y_true = [], []
    start_time = time.time()
    total_batches = len(test_loader)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

            # Display progress with time per batch
            progress = (batch_idx + 1) / total_batches * 100
            elapsed = time.time() - start_time
            print(f"\rProgress: {progress:.2f}% - Time per Batch: {elapsed / (batch_idx + 1):.2f}s", end="")

    # Calculate and display metrics
    elapsed_time = time.time() - start_time
    accuracy = accuracy_score(y_true, y_pred)
    print("\nTest Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=test_dataset.class_names))
    print(f"Total Time Elapsed: {elapsed_time:.2f} seconds")

    # Optional: Show some sample predictions for verification
    print("\nSample Predictions:")
    for i in range(min(5, len(y_true))):  # Show up to 5 predictions
        print(f"True Label: {test_dataset.class_names[y_true[i]]}, Predicted: {test_dataset.class_names[y_pred[i]]}")


# Run the evaluation
if __name__ == '__main__':
    evaluate_on_test('best_model_fold0.pth', 'data/xray_img/test', batch_size=32)
