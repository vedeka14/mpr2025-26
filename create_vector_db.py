# We need to import the function that creates our text chunks
from chunk_documents import process_and_chunk_documents

# Import from the new, recommended package
from langchain_huggingface import HuggingFaceEmbeddings

# We will use FAISS as our vector database
from langchain_community.vectorstores import FAISS

def create_and_save_vector_db(chunks):
    """
    Creates embeddings for text chunks and saves them to a FAISS vector database.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("➡️  Creating embeddings for the document chunks...")
    print("⏳ (This may take a few minutes as the model is downloaded for the first time)...")

    vector_db = FAISS.from_texts(texts=chunks, embedding=embedding_model)

    print("✅ Embeddings created successfully!")

    vector_db.save_local("faiss_index")

    print("✅ Vector database saved to 'faiss_index' folder.")

    return vector_db

# --- Main execution ---
if __name__ == "__main__":
    # The path to our extracted data folder
    extracted_data_path = "data/extracted_data/text/train"
    chunks = process_and_chunk_documents(extracted_data_path)

    # Check if any chunks were created before proceeding
    if chunks:
        # Now, create the vector database from these chunks
        create_and_save_vector_db(chunks)
    else:
        print("⚠️ No chunks were found. Please ensure your 'data/extracted_data/text/train' folder contains text files.")