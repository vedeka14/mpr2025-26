import tarfile
import os

def extract_tar_file(tar_path, output_path):
    """
    Extracts a .tar archive to a specified directory.
    """
    try:
        # Open the tar file
        with tarfile.open(tar_path, "r") as tar:
            print(f"Extracting all files from {tar_path}...")
            
            # Extract all contents to the specified path
            tar.extractall(path=output_path)
            
            print("Extraction complete.")
            
    except tarfile.TarError as e:
        print(f"Error extracting tar file: {e}")
    except FileNotFoundError:
        print(f"Error: The tar file at {tar_path} was not found.")

# --- Main part of the script ---
if __name__ == "__main__":
    # Point the script to the file you just downloaded.
    tar_file_path = "data/fc-amf-train-0000.tar"
    output_directory = "data/extracted_data"
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    extract_tar_file(tar_file_path, output_directory)
