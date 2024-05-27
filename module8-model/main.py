import logging
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Configuration variables
cover_path = r'C:\Users\alcui\workspace\mcse\module8\opdracht\opdracht\training-images\cover'
labels_path = r'C:\Users\alcui\workspace\mcse\module8\opdracht\opdracht\training-images\labels.csv'
batch_size = 8
epoch_size_encoder = 20
epoch_size_classifier = 20
quick_test = False

if quick_test:
    epoch_size_encoder = 1
    epoch_size_classifier = 1

max_test_images = 1000

# Configure logging
logging.basicConfig(level=logging.INFO)
os.system('')

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_encoder_output(self, x):
        return self.encoder(x)  # Return the output of the encoder

# Define the Classifier model
class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling layer
        self.fc = nn.Linear(64, hidden_size)  # Input size will be determined dynamically
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, 1)  # Output size changed to 1 for binary classification

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the encoder output
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)
        x = torch.sigmoid(x)  # Apply sigmoid activation for binary classification
        return x

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        label = int(self.df.iloc[idx, 1])
        return image, label

# Load the labels dataframe
df = pd.read_csv(labels_path)

# Convert class column to int
df['class'] = df['class'].astype(int)

train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create datasets and dataloaders for training and evaluation
train_dataset = CustomDataset(train_df, cover_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

eval_dataset = CustomDataset(eval_df, cover_path, transform=transform)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

# Initialize the autoencoder and move it to the device
autoencoder = Autoencoder().to(device)
logging.info("Autoencoder initialized")

# Define loss function and optimizer for autoencoder
criterion = nn.MSELoss()
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=0.001)

# Train the autoencoder
logging.info("Starting training of autoencoder")
for epoch in range(epoch_size_encoder):
    running_loss = 0.0
    processed_images = 0
    for i, (images, _) in enumerate(train_loader):
        if quick_test and processed_images >= max_test_images:
            break
        images = images.to(device)
        optimizer_ae.zero_grad()
        outputs = autoencoder(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer_ae.step()
        running_loss += loss.item()
        processed_images += images.size(0)
        if i % 100 == 99:
            logging.info(f"Epoch [{epoch+1}/{epoch_size_encoder}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / (i + 1):.4f}")
    logging.info(f"Epoch [{epoch+1}/{epoch_size_encoder}], Loss: {running_loss / len(train_loader):.4f}")

logging.info("Autoencoder training completed")

# Define the encoder part of the autoencoder
encoder = nn.Sequential(*list(autoencoder.encoder.children()))

# Determine the size of the flattened encoder output
with torch.no_grad():
    sample_input = torch.randn(1, 3, 64, 64).to(device)  # Assuming input images are 64x64
    sample_output = encoder(sample_input)
    flattened_size = sample_output.view(1, -1).size(1)

# Define the classifier
hidden_size = 32
num_classes = 1
classifier = Classifier(hidden_size).to(device)
logging.info("Classifier initialized")

# Define loss function and optimizer for classifier
criterion_cls = nn.BCELoss()
optimizer_cls = optim.Adam(classifier.parameters(), lr=0.001)

# Train the classifier
logging.info("Starting training of classifier")
for epoch in range(epoch_size_classifier):
    running_loss = 0.0
    processed_images = 0
    for i, (images, labels) in enumerate(train_loader):
        if quick_test and processed_images >= max_test_images:
            break
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        encoded = encoder(images)
        optimizer_cls.zero_grad()
        outputs = classifier(encoded)
        loss = criterion_cls(outputs, labels)
        loss.backward()
        optimizer_cls.step()
        running_loss += loss.item()
        processed_images += images.size(0)
        if i % 100 == 99:
            logging.info(f"Epoch [{epoch+1}/{epoch_size_classifier}], Batch [{i+1}/{len(train_loader)}], Loss: {running_loss / (i+1):.4f}")
    logging.info(f"Epoch [{epoch+1}/{epoch_size_classifier}], Loss: {running_loss / len(train_loader):.4f}")

logging.info("Classifier training completed")

# Evaluation
correct = 0
total = 0
processed_images = 0
all_predicted = []
all_labels = []
logging.info("Starting evaluation")
with torch.no_grad():
    for i, (images, labels) in enumerate(eval_loader):
        if quick_test and processed_images >= max_test_images:
            break
        images = images.to(device)
        labels = labels.to(device)
        encoded = encoder(images)
        outputs = classifier(encoded)
        predicted = (outputs > 0.5).float()
        all_predicted.extend(predicted.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        processed_images += images.size(0)

# Calculate Precision, Recall, and F1-score
precision = correct / total
recall = np.sum(all_labels == all_predicted) / np.sum(all_labels)
f1_score = 2 * (precision * recall) / (precision + recall)

logging.info(f"Accuracy: {100 * correct / total:.2f}%")
logging.info(f"Precision: {precision:.4f}")
logging.info(f"Recall: {recall:.4f}")
logging.info(f"F1 Score: {f1_score:.4f}")

# Calculate Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_predicted)
logging.info("Confusion Matrix:")
logging.info(conf_matrix)

# Save the entire autoencoder model
torch.save(autoencoder, 'autoencoder.pth')

# Save the entire classifier model
torch.save(classifier, 'classifier.pth')

# Save the state dictionary of the autoencoder
torch.save(autoencoder.state_dict(), 'autoencoder_state.pth')

# Save the state dictionary of the classifier
torch.save(classifier.state_dict(), 'classifier_state.pth')

# Save the state dictionary of the optimizer
torch.save(optimizer_cls.state_dict(), 'optimizer_state.pth')
