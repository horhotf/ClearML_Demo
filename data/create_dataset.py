import os
from pathlib import Path

from clearml import Dataset

DATA_ROOT = "./clearml-data/sample_data/images"

# # Create dataset
ds = Dataset.create(dataset_name="animal-dataset", dataset_project="clearml-data")

# # Add files
ds.add_files(path=DATA_ROOT)

# Count the amount of samples per class
root_folder = Path(DATA_ROOT)
counts = []
folders = sorted(os.listdir(root_folder))
for folder in folders:
    counts.append([len(os.listdir(root_folder / folder))])

# Log some dataset statistics
ds.get_logger().report_histogram(
    title="Dataset Statistics",
    series="Train Test Split",
    labels=folders,
    values=counts,
)

# Finalize and upload
# ds.upload()
# ds.finalize()
ds.finalize(auto_upload=True)
