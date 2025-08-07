import pandas as pd
import joblib
import numpy_financial as npf

# --- All the Agent Functions are unchanged ---

def extract_details_from_source(applicant_id, loan_df):
    """Simulates extracting key details for a specific applicant."""
    print(f"\n[Agent 1: Document Extraction]")
    applicant_data = loan_df[loan_df['id'] == applicant_id]
    if applicant_data.empty:
        print(f"-> Error: Applicant ID {applicant_id} not found.")
        return None
    
    details = applicant_data.iloc[0].to_dict()
    print(f"-> Success: Extracted details for applicant {applicant_id}.")
    return details

def fetch_credit_history(applicant_details):
    """Simulates fetching credit history."""
    print(f"\n[Agent 2: Credit History]")
    credit_info = {
        'fico_score': applicant_details.get('fico_range_low'),
        'loan_status_history': applicant_details.get('loan_status')
    }
    print(f"-> Success: Fetched FICO score: {credit_info['fico_score']}.")
    return credit_info

def assess_risk(applicant_details, risk_model):
    """Uses the pre-trained ML model to calculate a risk score."""
    print(f"\n[Agent 3: Risk Assessment]")
    try:
        features_for_model = [
            applicant_details['loan_amnt'],
            applicant_details['annual_inc'],
            applicant_details['dti'],
            applicant_details['fico_range_low']
        ]
        prediction_data = [features_for_model]
        risk_probability = risk_model.predict_proba(prediction_data)[0][1]
        print(f"-> Success: Calculated risk probability: {risk_probability:.2%}")
        return risk_probability
    except Exception as e:
        print(f"-> Error: Could not assess risk. Details: {e}")
        return None

def make_decision(risk_probability):
    """Makes a final decision based on the risk score and business rules."""
    print(f"\n[Agent 4: Decision]")
    if risk_probability is None:
        return 'Error', None

    if risk_probability > 0.5:
        decision = 'Rejected'
        reason = 'High risk score'
        print(f"-> Decision: {decision} (Reason: {reason})")
        return decision, reason
    elif risk_probability > 0.2:
        decision = 'Approved with Conditions'
        reason = 'Moderate risk score'
        print(f"-> Decision: {decision} (Reason: {reason})")
        return decision, reason
    else:
        decision = 'Approved'
        reason = 'Low risk score'
        print(f"-> Decision: {decision} (Reason: {reason})")
        return decision, reason

def calculate_emi(applicant_details, interest_rate=8.5):
    """Generates an EMI schedule if the loan is approved."""
    print(f"\n[Agent 5: EMI Calculation]")
    try:
        loan_amount = applicant_details['loan_amnt']
        term_in_months = int(str(applicant_details['term']).strip().split()[0])
        monthly_rate = (interest_rate / 100) / 12
        emi = -npf.pmt(rate=monthly_rate, nper=term_in_months, pv=loan_amount)
        print(f"-> Success: Calculated EMI is ${emi:,.2f}/month for {term_in_months} months.")
        return emi
    except Exception as e:
        print(f"-> Error: Could not calculate EMI. Details: {e}")
        return None

def generate_rejection_report(applicant_details, reason):
    """Provides a rejection reason and recommendations."""
    print(f"\n[Agent 6: Rejection Report]")
    report = (
        f"--- Loan Application Rejection ---\n"
        f"Applicant ID: {applicant_details['id']}\n"
        f"Reason for Rejection: {reason}.\n"
        f"Recommendation: We recommend improving your credit score and/or reducing your debt-to-income ratio before reapplying."
        f"\n----------------------------------"
    )
    print(report)
    return report

# --- Main Orchestrator ---
if __name__ == "__main__":
    print("--- Starting Automated Loan Application Processing ---")
    
    APPLICANT_ID_TO_PROCESS = 66310712 
    
    try:
        loan_df = pd.read_csv("dataset/loan.csv", low_memory=False)
        risk_model = joblib.load("risk_model.joblib")
        
        # --- NEW: Clean the ID column to fix the data type mismatch ---
        print("➡️  Ensuring ID column has the correct data type...")
        loan_df['id'] = pd.to_numeric(loan_df['id'], errors='coerce')
        loan_df.dropna(subset=['id'], inplace=True)
        loan_df['id'] = loan_df['id'].astype(int)
        print("✅ ID column cleaned.")
        # --- END OF NEW CODE ---

        print("✅ Models and data loaded successfully.")
    except FileNotFoundError:
        print("🔴 CRITICAL ERROR: Make sure 'dataset/loan.csv' and 'risk_model.joblib' exist.")
        exit()

    details = extract_details_from_source(APPLICANT_ID_TO_PROCESS, loan_df)
    
    if details:
        credit_history = fetch_credit_history(details)
        risk = assess_risk(details, risk_model)
        decision, reason = make_decision(risk)
        
        if decision.startswith('Approved'):
            calculate_emi(details)
        else:
            generate_rejection_report(details, reason)
    
    print("\n--- Workflow Complete ---")