import os
import json
from datasets import load_dataset

def download_and_save_dataset_samples():
    """
    Loads samples from the fc-amf-ocr dataset using the 'datasets' library
    and saves them as individual JSON files.
    """
    # The folder where we will save the documents
    save_dir = "data/extracted_data/text/train"
    os.makedirs(save_dir, exist_ok=True) # Ensure the directory exists

    print("➡️  Loading dataset 'lightonai/fc-amf-ocr' using the official library...")

    try:
        # CORRECTED: Changed name="small" to name="default"
        dataset = load_dataset("lightonai/fc-amf-ocr", name="default", split="train", streaming=True).take(3)

        print(f"✅ Dataset loaded. Now processing and saving 3 documents...")

        for i, sample in enumerate(dataset):
            # The 'text' field is a list of words; we join them into a single string.
            document_text = " ".join(sample['text'])

            # Use the sample's ID for a unique filename.
            filename = f"{sample['id']}.json"
            file_path = os.path.join(save_dir, filename)

            # Create a JSON structure similar to what our chunker expects
            output_data = {'text': document_text}

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)

            print(f"✅ Successfully saved document {i+1}: {filename}")

    except Exception as e:
        print(f"🔴 An error occurred: {e}")
        print("Please ensure you have an active internet connection.")

if __name__ == "__main__":
    download_and_save_dataset_samples()