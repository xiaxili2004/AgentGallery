import json
import os
import requests

def download_bill_pdfs(json_path: str, download_dir: str = "downloads"):
    """
    Extracts 'bill_content_url_eng' from a JSON file and downloads the files to the specified directory.
    Args:
        json_path (str): Path to the JSON file.
        download_dir (str): Directory to save the downloaded files.
    """
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract URLs and download files
    for entry in data.get("value", []):
        url = entry.get("bill_content_url_eng", "")
        if url:
            filename = os.path.join(download_dir, os.path.basename(url))
            if os.path.exists(filename):
                print(f"File already exists, skipping: {filename}")
                continue
            print(f"Downloading {url} -> {filename}")
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(filename, "wb") as out_file:
                    out_file.write(response.content)
            except Exception as e:
                print(f"Failed to download {url}: {e}")

    print("Download complete.")

# Example usage
if __name__ == "__main__":
    download_bill_pdfs("resources/legco_bill_index.json", "resources/legco")