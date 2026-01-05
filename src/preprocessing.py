

import pandas as pd

class DataPreprocessor:

    STRENGTH_MAPPING = {
        "weak": 0,
        "medium": 1,
        "strong": 2,
        "unbreakable": 3
    }

    def __init__(self, dataframe: pd.DataFrame):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame")

        self.df = dataframe.copy()

    def normalize_columns(self):
        self.df.columns = self.df.columns.str.strip().str.lower()
        print("Column names normalized")

    def remove_missing_values(self):
        initial_rows = len(self.df)
        self.df.dropna(inplace=True)
        print(f"Missing values removed: {initial_rows - len(self.df)}")

    def remove_duplicates(self):
        initial_rows = len(self.df)
        self.df.drop_duplicates(subset=["password"], inplace=True)
        print(f"Duplicate passwords removed: {initial_rows - len(self.df)}")

    def normalize_strength_labels(self, target_column="Strength"):

        self.df[target_column] = self.df[target_column].str.lower().str.strip()
        print("Strength labels normalized")

    def encode_target_column(self, target_column="Strength"):

        if target_column not in self.df.columns:
            raise ValueError(f"'{target_column}' column not found")

        unknown_labels = set(self.df[target_column]) - set(self.STRENGTH_MAPPING.keys())
        if unknown_labels:
            raise ValueError(f"Unknown Strength labels found: {unknown_labels}")

        self.df[target_column] = self.df[target_column].map(self.STRENGTH_MAPPING)

        print("Strength labels encoded safely")
        print("Encoding used:")
        for k, v in self.STRENGTH_MAPPING.items():
            print(f"   {k} -> {v}")

    def validate_columns(self, required_columns):
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        print("Required columns validated")

    def preprocess(self, target_column="Strength", required_columns=None):
        print("\nStarting preprocessing pipeline...\n")

        self.normalize_columns()

        if required_columns:
            required_columns = [c.lower() for c in required_columns]
            self.validate_columns(required_columns)

        self.remove_missing_values()
        self.remove_duplicates()
        self.normalize_strength_labels(target_column)
        self.encode_target_column(target_column)

        print("\nPreprocessing completed")
        print(f"Final dataset shape: {self.df.shape}")

        return self.df

    def save_cleaned_data(self, output_path):
        self.df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved at: {output_path}")



if __name__ == "__main__":
    from data_loader import DataLoader

    input_path = (
        "/home/kali/Music/hello/Password-Strength-Analyzer/"
        "dataset/raw/passwords_raw.csv"
    )
    output_path = (
        "/home/kali/Music/hello/Password-Strength-Analyzer/"
        "dataset/processed/passwords_cleaned.csv"
    )

    loader = DataLoader(input_path)
    df = loader.load_data()

    preprocessor = DataPreprocessor(df)
    cleaned_df = preprocessor.preprocess(
        target_column="strength",
        required_columns=["Password", "Strength"]
    )


    preprocessor.save_cleaned_data(output_path)

    print("\nSample Cleaned Data:")
    print(cleaned_df.head())
