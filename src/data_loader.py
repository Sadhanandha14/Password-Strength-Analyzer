"""
data_loader.py
----------------
This module is responsible for loading the password dataset
from a CSV file for training and evaluation purposes.

Author : (Your Name)
Project: Password Strength Analyzer using Machine Learning
"""


#   GOOD CODE 

# import os
# import pandas as pd


# class DataLoader:
#     """
#     DataLoader class handles loading and basic validation
#     of CSV datasets used in the project.
#     """

#     def __init__(self, file_path):
#         """
#         Constructor to initialize dataset file path

#         :param file_path: Path to the CSV dataset
#         """
#         self.file_path = file_path

#     def validate_file(self):
#         """
#         Checks whether the file exists and is a CSV file
#         """
#         if not os.path.exists(self.file_path):
#             raise FileNotFoundError(
#                 f"Dataset file not found at path: {self.file_path}"
#             )

#         if not self.file_path.endswith(".csv"):
#             raise ValueError("Invalid file format. Only CSV files are supported.")

#     def load_data(self):
#         """
#         Loads the CSV dataset into a Pandas DataFrame

#         :return: Pandas DataFrame
#         """
#         # Validate file before loading
#         self.validate_file()

#         try:
#             df = pd.read_csv(self.file_path)

#             if df.empty:
#                 raise ValueError("Loaded dataset is empty.")

#             print("✅ Dataset loaded successfully")
#             print(f"📊 Total records: {len(df)}")
#             print(f"📌 Columns: {list(df.columns)}")

#             return df

#         except pd.errors.ParserError:
#             raise ValueError("Error parsing the CSV file. Check file format.")

#         except Exception as e:
#             raise RuntimeError(f"Unexpected error while loading dataset: {e}")


# # -------------------- TESTING SECTION --------------------
# # This block runs only when this file is executed directly
# # It will NOT run when imported into other modules

# if __name__ == "__main__":
#     dataset_path = "/home/kali/Music/hello/Password-Strength-Analyzer/dataset/raw/passwords_raw.csv"

#     loader = DataLoader(dataset_path)
#     data = loader.load_data()

#     print("\n🔍 Sample Data:")
#     print(data.head())










"""
data_loader.py
----------------
This module loads and validates the password dataset
used for training the ML model behind the SecurePass AI UI.

Author : Pallavi
Project: Password Strength Analyzer using Machine Learning
"""

import os
import pandas as pd


class DataLoader:
    REQUIRED_COLUMNS = ["Password", "Strength"]
    def __init__(self, file_path: str):
        self.file_path = file_path
    def validate_file(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Dataset file not found: {self.file_path}"
            )

        if not self.file_path.lower().endswith(".csv"):
            raise ValueError("Invalid file format. Only CSV files are supported.")

    def validate_columns(self, df: pd.DataFrame):
        missing_cols = [
            col for col in self.REQUIRED_COLUMNS if col not in df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}"
            )

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["Password", "Strength"])
        df["Strength"] = df["Strength"].str.lower()
        df = df[df["Password"].str.len() > 0]
        return df

    def load_data(self) -> pd.DataFrame:
        self.validate_file()

        try:
            df = pd.read_csv(self.file_path)

            if df.empty:
                raise ValueError("Loaded dataset is empty.")

            self.validate_columns(df)
            df = self.clean_data(df)

            print("Dataset loaded successfully")
            print(f"Total records: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            print(f"Strength distribution:\n{df['Strength'].value_counts()}")

            return df

        except pd.errors.ParserError:
            raise ValueError("CSV parsing error. Check delimiter or format.")

        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")

if __name__ == "__main__":
    dataset_path = (
        "/home/kali/Music/hello/Password-Strength-Analyzer/"
        "dataset/raw/passwords_raw.csv"
    )

    loader = DataLoader(dataset_path)
    data = loader.load_data()

    print("\nSample Data:")
    print(data.head())








