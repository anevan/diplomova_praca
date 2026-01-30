import os
import pandas as pd


def load_csv(folder='datasets'):
    while True:
        try:
            # List available CSV files
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.endswith('.csv')]
                if files:
                    print("\nAvailable datasets:")
                    for i, f in enumerate(files, start=1):
                        print(f"{i}  {f}")
                else:
                    print(f"\nNo CSV files found in '{folder}'.")
                    files = []

            print("\nEnter the number to select a file from the list above,")
            print("or provide a full path to a CSV file.")
            user_input = input("Your choice: ").strip()

            # User selects from list
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(files):
                    path = os.path.join(folder, files[idx])
                else:
                    print("Invalid number. Please try again.")
                    continue
            else:
                path = user_input

            # Try to load the file
            df = pd.read_csv(path)
            filename = os.path.splitext(os.path.basename(path))[0]
            print(f"Loaded '{filename}.csv' successfully.")
            print(f"DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")

            # Print first 5 rows
            print("Preview of the first 5 rows:")
            print(df.head())
            return df, filename

        except Exception as e:
            print(f"Error: {e}. Please try again.")
