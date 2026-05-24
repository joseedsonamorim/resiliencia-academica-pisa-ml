import pandas as pd
import numpy as np
import gc
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_path: str = "project/data"):
        self.data_path = Path(data_path)
        self.logger = logging.getLogger(__name__)

    def load_with_memory_optimization(self, filepath: str, dtypes: dict = None,
                                     nrows: int = None) -> pd.DataFrame:
        """Load CSV with memory optimization (float32, int32)."""
        try:
            if dtypes is None:
                dtypes = {}

            df = pd.read_csv(
                filepath,
                dtype=dtypes,
                low_memory=True,
                nrows=nrows
            )

            # Convert floats to float32
            float_cols = df.select_dtypes(include=['float64']).columns
            for col in float_cols:
                df[col] = df[col].astype('float32')

            # Convert ints to int32 where possible
            int_cols = df.select_dtypes(include=['int64']).columns
            for col in int_cols:
                if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype('int32')

            gc.collect()
            self.logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df

        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {str(e)}")
            raise

    def load_stu_cog(self) -> pd.DataFrame:
        """Load cognitive dataset."""
        filepath = self.data_path / "CY08MSP_STU_COG_BRASIL.csv"
        return self.load_with_memory_optimization(str(filepath))

    def load_stu_qqq(self) -> pd.DataFrame:
        """Load questionnaire dataset."""
        filepath = self.data_path / "CY08MSP_STU_QQQ_BRASIL.csv"
        return self.load_with_memory_optimization(str(filepath))

    def merge_datasets(self, cog_df: pd.DataFrame, qqq_df: pd.DataFrame) -> pd.DataFrame:
        """Merge cognitive and questionnaire datasets on CNTSTUID."""
        try:
            merged = pd.merge(
                cog_df,
                qqq_df,
                on='CNTSTUID',
                how='inner'
            )

            self.logger.info(f"Merged dataset: {len(merged)} rows")
            gc.collect()
            return merged

        except Exception as e:
            self.logger.error(f"Error merging datasets: {str(e)}")
            raise

    def rename_crt_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename CRT OECD columns from PV*CRTH_NC to PV*CRT."""
        try:
            rename_dict = {}
            for i in range(1, 11):
                old_col = f"PV{i}CRTH_NC"
                new_col = f"PV{i}CRT"
                if old_col in df.columns:
                    rename_dict[old_col] = new_col

            if rename_dict:
                df = df.rename(columns=rename_dict)
                self.logger.info(f"Renamed {len(rename_dict)} CRT columns")

            return df

        except Exception as e:
            self.logger.error(f"Error renaming CRT columns: {str(e)}")
            raise

    def save_merged(self, df: pd.DataFrame, output_path: str = "project/outputs/tables/merged_pisa.csv"):
        """Save merged dataset."""
        try:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved merged dataset to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving merged dataset: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()
    print("DataLoader initialized successfully")
