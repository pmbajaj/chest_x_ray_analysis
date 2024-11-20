import os
import time  # Import time module to track elapsed time
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from models.model import XRayAnalysisModel
import torch.optim as optim
from sklearn.model_selection import KFold
import numpy as np

# Check for available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class XRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = os.listdir(root_dir)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        for cls_name in self.class_names:
            class_path = os.path.join(root_dir, cls_name, 'images')
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, img_file))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Main function
def main():
    dataset = XRayDataset(root_dir='data/xray_img/train', transform=transform)
    print("Number of training images found:", len(dataset))

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    accuracy_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f'Fold {fold + 1}/{k_folds}')

        # Create data loaders for this fold
        train_sampler = WeightedRandomSampler(weights=[1. / dataset.labels.count(i) for i in dataset.labels],
                                              num_samples=len(train_idx),
                                              replacement=True)
        train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_idx, num_workers=0)

        model = XRayAnalysisModel().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        best_loss = float('inf')
        patience = 5
        counter = 0
        num_epochs = 20

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            num_batches = len(train_loader)

            # Start time for the epoch
            epoch_start_time = time.time()

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Calculate and display progress percentage and elapsed time
                progress = (batch_idx + 1) / num_batches * 100
                elapsed_time = time.time() - epoch_start_time  # Calculate elapsed time
                print(f"\rEpoch [{epoch + 1}/{num_epochs}] - Fold [{fold + 1}/{k_folds}] - "
                      f"Progress: {progress:.2f}% - Elapsed Time: {elapsed_time:.2f}s",
                      end="")

            scheduler.step()

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("\nEarly stopping")
                    break

        # Evaluation on the validation set
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        accuracy_list.append(accuracy)
        print(f'\nFold {fold + 1} Accuracy: {accuracy:.4f}')
        print(classification_report(all_labels, all_preds, target_names=dataset.class_names))

    print(f'Average K-Fold Accuracy: {np.mean(accuracy_list):.4f}')


# Run the main function
if __name__ == '__main__':
    main()
