import urllib.request
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# URLs to download
urls = [
    "https://github.com/flame0409/Jane-Street-Market-Prediction/raw/master/example_test.csv",
    "https://github.com/flame0409/Jane-Street-Market-Prediction/raw/master/example_train.csv"
]

# Download files
for url in urls:
    filename = url.split('/')[-1]
    output_path = os.path.join('data', filename)
    print(f"Downloading {url} to {output_path}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

print("Download process completed.") 