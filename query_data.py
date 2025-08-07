import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# --- IMPORTANT: API Key Setup ---
# You need a Groq API key for this to work.
# 1. Go to https://console.groq.com/keys
# 2. Create a free account and generate an API key.
# 3. Set it as an environment variable named "GROQ_API_KEY".
#    In your terminal (for one session):
#    Windows: set GROQ_API_KEY=YOUR_API_KEY_HERE
#    macOS/Linux: export GROQ_API_KEY=YOUR_API_KEY_HERE

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    """Main function to load the database and answer questions."""
    # Check if the API key is set
    if "GROQ_API_KEY" not in os.environ:
        print("Error: GROQ_API_KEY environment variable not set.")
        print("Please get a free API key from https://console.groq.com/keys and set it.")
        return

    # Load the vector database
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever()

    # Define our prompt template
    template = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Define the LLM
    llm = ChatGroq(model_name="llama3-8b-8192")

    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ Your RAG system is ready. Ask a question about your document.")
    print("   Type 'exit' to quit.")

    # Loop to ask questions
    while True:
        question = input("\nYour Question: ")
        if question.lower() == 'exit':
            break
        
        # Get the answer from the RAG chain
        answer = rag_chain.invoke(question)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()