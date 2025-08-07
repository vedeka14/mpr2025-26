import pandas as pd
import os

# Define file paths
CSV_PATH = "dataset/loan.csv"
OUTPUT_DIR = "data/extracted_data/text/train"

def create_documents_from_csv_in_chunks(csv_path, output_dir, num_documents=500, chunk_size=100000):
    """
    Reads the large loan CSV in smaller chunks to conserve memory,
    then creates a text file for each of the first N loans.
    """
    print("➡️  Processing loan data from CSV in memory-efficient chunks...")
    try:
        # Columns we want to keep
        columns_to_include = [
            'id', 'loan_amnt', 'term', 'grade', 'emp_length',
            'home_ownership', 'annual_inc', 'purpose', 'dti',
            'fico_range_low', 'loan_status'
        ]
        
        # Create an iterator to read the CSV in chunks
        chunk_iter = pd.read_csv(
            csv_path,
            usecols=columns_to_include,
            chunksize=chunk_size,
            low_memory=False
        )
        
        # Create an empty DataFrame to hold our samples
        sample_df = pd.DataFrame()
        
        # Loop through the chunks until we have enough samples
        for chunk in chunk_iter:
            sample_df = pd.concat([sample_df, chunk])
            if len(sample_df) >= num_documents:
                sample_df = sample_df.head(num_documents)
                break
        
        print(f"✅ Loaded the first {len(sample_df)} rows for processing.")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"➡️  Generating {num_documents} text files...")

        for index, row in sample_df.iterrows():
            if pd.isna(row['id']):
                continue
            
            file_name = f"loan_app_{int(row['id'])}.txt"
            file_path = os.path.join(output_dir, file_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("--- Loan Application Summary ---\n\n")
                f.write(f"Application ID: {int(row['id'])}\n")
                f.write(f"Loan Amount: ${row.get('loan_amnt', 0):,.2f}\n")
                f.write(f"Loan Term: {row.get('term', 'N/A')}\n")
                f.write(f"Loan Grade: {row.get('grade', 'N/A')}\n")
                f.write(f"Employment Length: {row.get('emp_length', 'N/A')}\n")
                f.write(f"Home Ownership: {row.get('home_ownership', 'N/A')}\n")
                f.write(f"Annual Income: ${row.get('annual_inc', 0):,.2f}\n")
                f.write(f"Debt-to-Income Ratio: {row.get('dti', 'N/A')}\n")
                f.write(f"FICO Score (low): {row.get('fico_range_low', 'N/A')}\n")
                f.write(f"Loan Status: {row.get('loan_status', 'N/A')}\n")
            
        print(f"✅ Created {len(os.listdir(output_dir))} documents in the '{output_dir}' folder.")

    except FileNotFoundError:
        print(f"🔴 ERROR: The file was not found at {csv_path}")
    except Exception as e:
        print(f"🔴 An error occurred: {e}")

if __name__ == "__main__":
    print(f"Cleaning old files from '{OUTPUT_DIR}'...")
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            os.remove(os.path.join(OUTPUT_DIR, file))
    
    create_documents_from_csv_in_chunks(CSV_PATH, OUTPUT_DIR)