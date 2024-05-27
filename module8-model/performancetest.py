from openpyxl import Workbook
from openpyxl.chart import ScatterChart, Reference, Series
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

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

# Custom dataset class for test images
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f.lower().endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

# Define transforms for normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Paths to the saved models
autoencoder_state_path = 'autoencoder_state.pth'
classifier_state_path = 'classifier_state.pth'
optimizer_state_path = 'optimizer_state.pth'  # Optional, if you need to load optimizer state

# Path to the test images directory
test_images_path = r'C:\Users\alcui\workspace\mcse\module8\opdracht\opdracht\test-images'
warmup_image_path = r'C:\Users\alcui\workspace\mcse\module8\opdracht\opdracht\warmup.jpg'

# Define batch size
batch_size = 1  # Process images one by one

# Create test dataset and dataloader
test_dataset = TestDataset(test_images_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Path to save the Excel file
excel_file_path = 'inference_times.xlsx'

# Create Excel workbook
workbook = Workbook()
worksheet = workbook.active
worksheet.title = "Inference Times"
worksheet.append(['Filename', 'File Size (KB)', 'Inference Time (milliseconds)', 'Device'])

# Evaluation
logging.info("Starting evaluation on test images")

# Iterate over CPU and CUDA (if available)
for device_name, device_type in [('cpu', torch.device('cpu')), ('cuda', torch.device('cuda')) if torch.cuda.is_available() else ('cpu', torch.device('cpu'))]:
    # Load models and optimizer for the current device
    autoencoder = Autoencoder().to(device_type)
    classifier = Classifier(hidden_size=32).to(device_type)  # Match hidden_size to the trained model

    autoencoder.load_state_dict(torch.load(autoencoder_state_path, map_location=device_type))
    classifier.load_state_dict(torch.load(classifier_state_path, map_location=device_type))

    autoencoder.eval()

    classifier.eval()

    # Optionally, recreate the optimizer and load its state dictionary
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=0.001)
    optimizer_cls.load_state_dict(torch.load(optimizer_state_path, map_location=device_type))

    # Do warmup
    specific_image = Image.open(warmup_image_path).convert('RGB')
    specific_image_tensor = transform(specific_image).unsqueeze(0)  # Add batch dimension
    specific_image_tensor = specific_image_tensor.to(device_type)  # Move to device

    with torch.no_grad():
        # Measure the start time
        start_time = time.time_ns()
        encoded = autoencoder.encoder(specific_image_tensor)
        outputs = classifier(encoded)
        # Measure the end time
        end_time = time.time_ns()
        inference_time = (end_time - start_time) / 1000_000  # Convert to milliseconds
        logging.info(f"Warmup Image: Inference Time: {inference_time:.4f} milliseconds, Device: {device_name}")

    # Get sorted test images by file size
    sorted_test_loader = sorted(test_loader, key=lambda x: os.path.getsize(x[1][0]))

    for images, image_names in sorted_test_loader:
        images = images.to(device_type)

        # Measure the start time
        start_time = time.time_ns()

        encoded = autoencoder.encoder(images)
        outputs = classifier(encoded)

        # Measure the end time
        end_time = time.time_ns()

        inference_time = (end_time - start_time) / 1000_000  # Convert to milliseconds

        # Get file size
        file_size_kb = os.path.getsize(image_names[0]) / 1024

        # Log the filename, file size, inference time, and device used
        logging.info(f"Image: {image_names[0]}, File Size: {file_size_kb:.2f} KB, Inference Time: {inference_time:.4f} milliseconds, Device: {device_name}")

        # Write to Excel worksheet
        worksheet.append([image_names[0], file_size_kb, inference_time, device_name])

# Add a chart to the Excel worksheet
chart = ScatterChart()
chart.title = "Inference Times vs. File Size"
chart.x_axis.title = "File Size (KB)"
chart.y_axis.title = "Inference Time (milliseconds)"

# Add CPU data to the chart
cpu_times = Reference(worksheet, min_col=3, max_col=3, min_row=2, max_row=len(test_dataset) + 1)
cpu_sizes = Reference(worksheet, min_col=2, max_col=2, min_row=2, max_row=len(test_dataset) + 1)
cpu_series = Series(cpu_times, xvalues=cpu_sizes, title="CPU")
chart.series.append(cpu_series)

# Add CUDA data to the chart
cuda_times = Reference(worksheet, min_col=3, max_col=3, min_row=len(test_dataset) + 3, max_row=2 * len(test_dataset) + 2)
cuda_sizes = Reference(worksheet, min_col=2, max_col=2, min_row=len(test_dataset) + 3, max_row=2 * len(test_dataset) + 2)
cuda_series = Series(cuda_times, xvalues=cuda_sizes, title="CUDA")
chart.series.append(cuda_series)

# Add the chart to the worksheet
worksheet.add_chart(chart, "E2")

# Save the Excel workbook
workbook.save(excel_file_path)

