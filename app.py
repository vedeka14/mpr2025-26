import streamlit as st
import pandas as pd
import joblib
import numpy_financial as npf
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Loan Processor",
    page_icon="🤖",
    layout="wide"
)

# --- Caching Models and Data for Performance ---
@st.cache_resource
def load_models_and_data():
    """Loads all necessary models and data once and caches them."""
    print("Loading models and data...")
    # Load Risk Assessment Model and Data
    risk_model = joblib.load("risk_model.joblib")
    loan_df = pd.read_csv("dataset/loan.csv", low_memory=False)
    # Clean the ID column to fix the data type mismatch
    loan_df['id'] = pd.to_numeric(loan_df['id'], errors='coerce')
    loan_df.dropna(subset=['id'], inplace=True)
    loan_df['id'] = loan_df['id'].astype(int)
    
    # Load RAG Components
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    
    print("Loading complete.")
    return risk_model, loan_df, vector_db, embedding_model

# --- Agent Functions (Copied from loan_processor.py) ---
# We include the agent logic directly in our app for simplicity.
def assess_risk(applicant_details, risk_model):
    features = ['loan_amnt', 'annual_inc', 'dti', 'fico_range_low']
    applicant_features = pd.DataFrame([applicant_details])[features]
    risk_probability = risk_model.predict_proba(applicant_features)[0][1]
    return risk_probability

def make_decision(risk_probability):
    if risk_probability > 0.5: return 'Rejected', 'High risk score'
    elif risk_probability > 0.2: return 'Approved with Conditions', 'Moderate risk score'
    else: return 'Approved', 'Low risk score'

def calculate_emi(applicant_details, interest_rate=8.5):
    loan_amount = applicant_details['loan_amnt']
    term_in_months = int(str(applicant_details['term']).strip().split()[0])
    monthly_rate = (interest_rate / 100) / 12
    emi = -npf.pmt(rate=monthly_rate, nper=term_in_months, pv=loan_amount)
    return emi

# --- Main Application UI ---
st.title("🤖 RAG-Enabled Loan Application & Risk Assessment System")

# Load all necessary components
try:
    risk_model, loan_df, vector_db, embedding_model = load_models_and_data()
    
    # --- Sidebar for Mode Selection ---
    st.sidebar.title("Select Mode")
    app_mode = st.sidebar.radio(
        "Choose the system's function:",
        ("Loan Risk Assessment", "Query Documents (RAG)")
    )

    if app_mode == "Loan Risk Assessment":
        st.header("Loan Application Risk Assessment")
        
        # Get a list of applicant IDs to choose from
        applicant_ids = loan_df['id'].head(100).tolist() # Use first 100 for performance
        selected_id = st.selectbox("Select an Applicant ID to Assess:", applicant_ids)

        if st.button("Assess Risk"):
            with st.spinner("Running workflow..."):
                # --- Run the Agent Workflow ---
                applicant_details = loan_df[loan_df['id'] == selected_id].iloc[0].to_dict()
                
                # Agent 3: Risk Assessment
                risk_prob = assess_risk(applicant_details, risk_model)
                
                # Agent 4: Decision
                decision, reason = make_decision(risk_prob)

                st.subheader(f"Assessment for Applicant ID: {selected_id}")
                
                if decision.startswith('Approved'):
                    st.success(f"Decision: {decision} (Reason: {reason})")
                    st.metric(label="Calculated Risk Probability", value=f"{risk_prob:.2%}")

                    # Agent 5: EMI Calculation
                    emi = calculate_emi(applicant_details)
                    st.info(f"Generated EMI: **${emi:,.2f} / month**")
                else:
                    st.error(f"Decision: {decision} (Reason: {reason})")
                    st.metric(label="Calculated Risk Probability", value=f"{risk_prob:.2%}")

                    # Agent 6: Rejection Report
                    st.warning(
                        "**Recommendation:** We recommend improving the applicant's credit score and/or reducing their debt-to-income ratio before reapplying."
                    )

    elif app_mode == "Query Documents (RAG)":
        st.header("Query Loan Documents with RAG")
        
        # Setup RAG chain
        retriever = vector_db.as_retriever()
        template = "Context: {context}\nQuestion: {question}\nAnswer:"
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGroq(model_name="llama3-8b-8192")
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        user_question = st.text_input("Ask a question about the first 5 loan applications:")
        
        if user_question:
            # Check for API key
            if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
                st.error("GROQ_API_KEY is not set. Please set it in your terminal before running the app.")
            else:
                with st.spinner("Searching for answers..."):
                    response = rag_chain.invoke(user_question)
                    st.markdown(response)

except FileNotFoundError as e:
    st.error(f"🔴 CRITICAL ERROR: A required file was not found.")
    st.error(f"Details: {e}")
    st.info("Please make sure 'dataset/loan.csv', 'risk_model.joblib', and the 'faiss_index' folder are all present in your project directory.")