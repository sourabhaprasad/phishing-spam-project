# Create a file named download_dataset.py
import kagglehub

# Download dataset
path = kagglehub.dataset_download("meruvulikith/190k-spam-ham-email-dataset-for-classification")
print("Path to dataset files:", path)