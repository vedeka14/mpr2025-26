import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_and_chunk_documents(extracted_data_path):
    """
    Reads text and JSON files, extracts text, and chunks it into smaller pieces.
    """
    all_text = ""
    # Go through all files in the directory
    for filename in os.listdir(extracted_data_path):
        file_path = os.path.join(extracted_data_path, filename)

        # If it's a JSON file, load it and extract text from the 'text' key
        if filename.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'text' in data:
                    all_text += data['text'] + "\n"

        # If it's a text file, just read its content directly
        elif filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n"

    # Define our text chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Use the text splitter to create our chunks
    text_chunks = text_splitter.split_text(all_text)

    print(f"Original document text size: {len(all_text)} characters")
    print(f"Created {len(text_chunks)} text chunks.")

    return text_chunks

# --- Main part of the script ---
# This part only runs if you execute this file directly
if __name__ == "__main__":
    # The path to our extracted data folder
    extracted_text_path = "data/extracted_data/text/train"

    # Run the function to chunk our documents
    chunks = process_and_chunk_documents(extracted_text_path)

    # Print the first few chunks to see what they look like
    print("\n--- Sample Chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)