# In your download_data.py file, make sure it looks exactly like this

import os
from huggingface_hub import hf_hub_download

# The name of the online folder we are downloading from.
repo_id = "lightonai/fc-amf-ocr"

# The name of the specific file inside that folder.
# We'll try a known file from the dataset's documentation.

# In your download_data.py file
# Change the filename to this:
filename = "fc-amf-train-0002.tar"
# The name of the folder on your computer where it will be saved.
local_dir = "./data"

# The rest of this code will run automatically.
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

print(f"Downloading {filename} from {repo_id}...")

downloaded_file = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)

print(f"File downloaded to: {downloaded_file}")