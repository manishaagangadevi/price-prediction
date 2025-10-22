# src/utils.py

import requests
from tqdm import tqdm
import os

def download_images(images_to_download, save_directory):
    """
    Downloads images one by one from a list of (URL, ID) tuples.
    This is a simpler, more reliable version than the original.
    """
    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"Starting sequential download of {len(images_to_download)} images...")
    
    for url, sample_id in tqdm(images_to_download):
        file_path = os.path.join(save_directory, f"{sample_id}.jpg")
        
        # Only download if the file doesn't already exist
        if not os.path.exists(file_path):
            try:
                response = requests.get(url, timeout=10) # 10-second timeout
                response.raise_for_status() # Raise an exception for bad status codes (404, 500, etc.)
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                # This will catch connection errors, timeouts, bad status codes, etc.
                # print(f"Could not download {url}: {e}")
                pass # We'll just skip broken links silently
                
    print("Image download process complete.")