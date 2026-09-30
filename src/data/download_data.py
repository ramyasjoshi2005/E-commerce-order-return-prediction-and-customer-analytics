import kagglehub
import os
import shutil

def download_superstore_dataset():
    print("Downloading Global Superstore dataset...")
    # Download latest version
    path = kagglehub.dataset_download("apoorvaappz/global-super-store-dataset")
    
    # Destination directory
    dest_dir = "data/raw"
    os.makedirs(dest_dir, exist_ok=True)
    
    # Move files to data/raw
    for filename in os.listdir(path):
        source = os.path.join(path, filename)
        destination = os.path.join(dest_dir, filename)
        shutil.copy2(source, destination)
        print(f"Copied {filename} to {dest_dir}")
        
if __name__ == "__main__":
    download_superstore_dataset()
