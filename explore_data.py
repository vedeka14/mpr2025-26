import pandas as pd

# Define the path to our new dataset file
DATASET_PATH = "dataset/loan.csv"

def explore_loan_data(path):
    """
    Loads the loan.csv file and prints a summary.
    """
    print(f"➡️  Loading data from {path}...")
    try:
        # Load the CSV file into a Pandas DataFrame
        # low_memory=False is used to handle the large file with mixed data types.
        df = pd.read_csv(path, low_memory=False)
        print("✅ Data loaded successfully!")
        
        # --- Data Exploration ---
        print("\n--- First 5 Rows of the Dataset ---")
        print(df.head())

        print("\n\n--- Dataset Info (Columns, Data Types) ---")
        df.info()

    except FileNotFoundError:
        print(f"🔴 ERROR: The file was not found at {path}")
        print("Please make sure the 'loan.csv' file is inside the 'dataset' folder.")
    except Exception as e:
        print(f"🔴 An error occurred: {e}")

if __name__ == "__main__":
    explore_loan_data(DATASET_PATH)