import kagglehub

# Download latest version
path = kagglehub.dataset_download("realtimear/hand-wash-dataset")

print("Path to dataset files:", path)