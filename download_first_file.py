import os
from huggingface_hub import list_repo_files, hf_hub_download

def find_first_tar_file(repo_id):
    """
    Finds the first .tar file in a Hugging Face dataset by listing all files.

    Args:
        repo_id (str): The ID of the repository on Hugging Face Hub.

    Returns:
        str or None: The filename of the first .tar file found, or None if not found.
    """
    print(f"Searching for a .tar file in repository: {repo_id}...")
    try:
        # We specify repo_type="dataset" to make sure we're looking in the right place.
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        
        # We'll look for the first .tar file in the list.
        for filename in files:
            if filename.endswith(".tar"):
                print(f"Found a .tar file: {filename}")
                return filename
                
    except Exception as e:
        print(f"An error occurred while listing files: {e}")
    
    print("No .tar files found in the repository.")
    return None

def download_file_from_repo(repo_id, filename, local_dir):
    """
    Downloads a specific file from a Hugging Face repository.

    Args:
        repo_id (str): The ID of the repository on Hugging Face Hub.
        filename (str): The specific file path to download.
        local_dir (str): The local directory to save the file in.
    """
    if filename is None:
        print("Cannot download, no file was specified.")
        return
        
    print(f"Attempting to download {filename} from {repo_id}...")

    try:
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",  # We specify repo_type again for the download.
            local_dir=local_dir
        )
        print(f"Success! File downloaded to: {downloaded_file}")
    except Exception as e:
        print(f"An error occurred during download: {e}")

# --- Main part of the script ---
if __name__ == "__main__":
    repo_id = "lightonai/fc-amf-ocr"
    local_dir = "./data"
    
    # First, find a valid file path by listing the repository contents.
    file_to_download = find_first_tar_file(repo_id)
    
    # Then, if a file was found, download it.
    download_file_from_repo(repo_id, file_to_download, local_dir)

