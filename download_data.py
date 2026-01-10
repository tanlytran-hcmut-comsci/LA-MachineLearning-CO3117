import kagglehub

# Download image dataset
# path = kagglehub.dataset_download("preatcher/standard-ocr-dataset")

# Download EMNIST dataset
path = kagglehub.dataset_download("crawford/emnist")
print("Path to dataset files:", path)