

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








