import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib # Used to save our trained model

# Define the path to our dataset file
DATASET_PATH = "dataset/loan.csv"
MODEL_FILE_PATH = "risk_model.joblib"

def train_risk_model(path):
    """
    Loads loan data, preprocesses it, trains a classification model,
    evaluates it, and saves the trained model to a file.
    """
    print(f"➡️  Loading data from {path}...")
    try:
        df = pd.read_csv(path, low_memory=False)
        print("✅ Data loaded successfully!")

        # --- 1. Data Preprocessing and Feature Selection ---
        print("➡️  Preprocessing data...")
        
        # We'll select a few key features for our model
        features = [
            'loan_amnt',      # The amount of the loan
            'annual_inc',     # The borrower's annual income
            'dti',            # Debt-to-Income ratio
            'fico_range_low'  # The borrower's FICO credit score
        ]
        
        # The 'loan_status' column is our target variable
        target = 'loan_status'
        
        # Create a simplified dataframe with only the columns we need
        model_df = df[features + [target]].copy()
        
        # Drop rows with missing values in our key columns
        model_df.dropna(inplace=True)
        
        # Create our target variable: 1 for risky loans, 0 for good loans
        # We consider 'Charged Off', 'Default', etc., as risky.
        risky_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'Late (31-120 days)']
        model_df['is_risky'] = model_df['loan_status'].isin(risky_statuses).astype(int)
        
        # Define our features (X) and target (y)
        X = model_df[features]
        y = model_df['is_risky']
        print("✅ Data preprocessed.")

        # --- 2. Train/Test Split ---
        # We'll use 80% of the data for training and 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # --- 3. Train the Model ---
        print("➡️  Training the Logistic Regression model...")
        # Logistic Regression is a simple, effective, and fast model for this task
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        print("✅ Model trained.")

        # --- 4. Evaluate the Model ---
        print("➡️  Evaluating model performance...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Model Accuracy: {accuracy:.2%}")
        # A confusion matrix shows us how many predictions were right/wrong
        print("\n--- Confusion Matrix ---")
        print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                         columns=['Predicted Good', 'Predicted Risky'],
                         index=['Actual Good', 'Actual Risky']))

        # --- 5. Save the Trained Model ---
        print(f"\n➡️  Saving trained model to {MODEL_FILE_PATH}...")
        joblib.dump(model, MODEL_FILE_PATH)
        print("✅ Model saved successfully.")

    except FileNotFoundError:
        print(f"🔴 ERROR: The file was not found at {path}")
    except Exception as e:
        print(f"🔴 An error occurred: {e}")

if __name__ == "__main__":
    train_risk_model(DATASET_PATH)