from huggingface_hub import list_repo_files

# The name of the online folder we are downloading from.
repo_id = "lightonai/fc-amf-ocr"

print(f"Listing files in repository: {repo_id}")

try:
    # We must add repo_type="dataset" to tell the function to look for a dataset,
    # as it defaults to looking for a model.
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")

    # Print each file path on a new line.
    for filename in files:
        print(filename)
except Exception as e:
    print(f"An error occurred: {e}")