import pandas as pd
import os

# Define file paths
CSV_PATH = "dataset/loan.csv"
OUTPUT_DIR = "data/extracted_data/text/train"

def create_documents_from_csv(csv_path, output_dir, num_documents=5):
    """
    Reads the loan data CSV, selects a few loans, and creates
    a text file for each one summarizing its key details.
    """
    print("➡️  Loading loan data from CSV...")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"✅ Loaded {len(df)} rows of data.")

        # --- Select a few important columns for our documents ---
        # We choose columns that would typically appear on a loan application.
        columns_to_include = [
            'id', 'loan_amnt', 'term', 'grade', 'emp_length',
            'home_ownership', 'annual_inc', 'purpose', 'dti',
            'fico_range_low', 'loan_status'
        ]
        
        # Take the first few rows as our samples
        sample_df = df[columns_to_include].head(num_documents)
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"➡️  Generating {num_documents} text files...")

        # Loop through each sample loan and create a text file
        for index, row in sample_df.iterrows():
            file_name = f"loan_app_{int(row['id'])}.txt"
            file_path = os.path.join(output_dir, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("--- Loan Application Summary ---\n\n")
                f.write(f"Application ID: {int(row['id'])}\n")
                f.write(f"Loan Amount: ${row['loan_amnt']:,.2f}\n")
                f.write(f"Loan Term: {row['term']}\n")
                f.write(f"Loan Grade: {row['grade']}\n")
                f.write(f"Employment Length: {row['emp_length']}\n")
                f.write(f"Home Ownership: {row['home_ownership']}\n")
                f.write(f"Annual Income: ${row['annual_inc']:,.2f}\n")
                f.write(f"Debt-to-Income Ratio: {row['dti']}\n")
                f.write(f"FICO Score (low): {row['fico_range_low']}\n")
                f.write(f"Loan Status: {row['loan_status']}\n")
            
            print(f"✅ Created document: {file_name}")

    except FileNotFoundError:
        print(f"🔴 ERROR: The file was not found at {csv_path}")
    except Exception as e:
        print(f"🔴 An error occurred: {e}")

if __name__ == "__main__":
    # First, let's clean the directory of any old files
    for file in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, file))
    
    create_documents_from_csv(CSV_PATH, OUTPUT_DIR)