import os
import sys
import subprocess
import zipfile
import pandas as pd

def setup_kaggle_api():
    """
    Check if kaggle API is installed and configured
    """
    try:
        import kaggle
        print("Kaggle API already installed")
    except ImportError:
        print("Installing Kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    
    # Check if kaggle.json exists
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_api_path = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_api_path):
        print("\nKaggle API credentials not found.")
        print("Please visit https://www.kaggle.com/account and create an API token.")
        print("Then, place the downloaded kaggle.json file in ~/.kaggle/ directory.")
        print("On Windows, this is typically: C:\\Users\\<username>\\.kaggle\\")
        
        username = input("Enter your Kaggle username: ")
        key = input("Enter your Kaggle API key: ")
        
        # Create kaggle directory if it doesn't exist
        os.makedirs(kaggle_dir, exist_ok=True)
        
        # Create kaggle.json file
        with open(kaggle_api_path, 'w') as f:
            f.write('{{"username":"{0}","key":"{1}"}}'.format(username, key))
        
        # Set permissions
        os.chmod(kaggle_api_path, 0o600)
        
        print("Kaggle API credentials saved.")
    else:
        print("Kaggle API credentials found.")

def download_jane_street_dataset(data_dir='data'):
    """
    Download the Jane Street Market Prediction dataset from Kaggle
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the dataset
    print("\nDownloading Jane Street Market Prediction dataset...")
    try:
        command = ["kaggle", "competitions", "download", "-c", "jane-street-market-prediction", "-p", data_dir]
        subprocess.check_call(command)
        
        # Extract the dataset
        print("\nExtracting dataset...")
        zip_path = os.path.join(data_dir, "jane-street-market-prediction.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Remove zip file after extraction
            os.remove(zip_path)
            print("Dataset extracted successfully.")
        else:
            print(f"Error: Could not find zip file at {zip_path}")
            
        # Show dataset contents
        print("\nDataset contents:")
        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                print(f"  {file} ({file_size:.2f} MB)")
        
        # Check sample of the data
        train_path = os.path.join(data_dir, "train.csv")
        if os.path.exists(train_path):
            print("\nSample of training data:")
            train_sample = pd.read_csv(train_path, nrows=5)
            print(train_sample.head())
            
            # Print column names
            print("\nColumns in the dataset:")
            for col in train_sample.columns:
                print(f"  {col}")
                
            # Print shape of the dataset
            print(f"\nShape of the training data: {train_sample.shape}")
        
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    setup_kaggle_api()
    download_jane_street_dataset() 