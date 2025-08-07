import os
import joblib
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

def list_available_docs(path="data/extracted_data/text/train"):
    """Lists the application IDs from the filenames in the directory."""
    try:
        ids = [f.split('_')[-1].split('.')[0] for f in os.listdir(path) if f.startswith("loan_app")]
        return ids
    except FileNotFoundError:
        return []

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    """Main function to load the database and answer questions."""
    if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
        print("🔴 ERROR: GROQ_API_KEY is not set.")
        print("Please get a free API key from https://console.groq.com/keys and set it.")
        return

    # --- Load the Risk Model and the full CSV dataset ---
    print("➡️  Loading Risk Assessment Model...")
    try:
        risk_model = joblib.load("risk_model.joblib")
        print("✅ Risk Model loaded.")
        
        print("➡️  Loading full loan dataset for lookups...")
        loan_df = pd.read_csv("dataset/loan.csv", low_memory=False)
        print("✅ Full dataset loaded.")
        
    except FileNotFoundError:
        print("🔴 ERROR: Could not find 'risk_model.joblib' or 'dataset/loan.csv'.")
        print("Please make sure you have run the training script and the data is in place.")
        return
    # --- END OF NEW SECTION ---

    print("➡️  Loading RAG system...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever()
    print("✅ RAG system loaded.")
    
    available_ids = list_available_docs()
    if available_ids:
        print("\n✅ System ready. I have knowledge of the following applications:")
        print(f"   {', '.join(available_ids)}")
    
    print("\n   Type a question (e.g., 'What is the loan status for 68407277?')")
    print("   OR 'assess risk for [ID]'")
    print("   OR 'exit' to quit.")

    # --- RAG Chain definition ---
    template = """
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Keep the answer concise.

    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model_name="llama3-8b-8192")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    while True:
        question = input("\nYour Command: ")
        if question.lower() == 'exit':
            break

        # --- Logic to switch between RAG and Risk Assessment ---
        if question.lower().startswith("assess risk for"):
            try:
                # Extract the ID from the command
                app_id = int(question.split()[-1])
                
                # Find the applicant's data in the dataframe
                applicant_data = loan_df[loan_df['id'] == app_id]
                
                if applicant_data.empty:
                    print("\nAnswer: Applicant ID not found in the dataset.")
                    continue
                
                # Select the features our model needs
                features = ['loan_amnt', 'annual_inc', 'dti', 'fico_range_low']
                applicant_features = applicant_data[features].dropna()

                if applicant_features.empty:
                    print("\nAnswer: Applicant data is missing key features for assessment.")
                    continue

                # Get the prediction and probability
                prediction = risk_model.predict(applicant_features)[0]
                probability = risk_model.predict_proba(applicant_features)[0]
                
                # Print the report
                print("\n--- Risk Assessment Report ---")
                print(f"Applicant ID: {app_id}")
                if prediction == 0:
                    print("Prediction: Good Risk ✅")
                    print(f"Confidence: {probability[0]:.2%}")
                else:
                    print("Prediction: High Risk 🔴")
                    print(f"Confidence: {probability[1]:.2%}")
                print("-----------------------------")

            except (ValueError, IndexError):
                print("\nInvalid command. Please use the format: assess risk for [ID]")
        else:
            # If not an assess command, use the RAG chain
            answer = rag_chain.invoke(question)
            print("\nAnswer:", answer)
        # --- END OF NEW LOGIC ---

if __name__ == "__main__":
    main()