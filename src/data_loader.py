"""DataLoader — Load, combine, and clean YouTube trending CSV files."""

import os

import pandas as pd


class DataLoader:
    """Load, combine, and clean YouTube trending CSV files."""

    ENGLISH_MARKETS = {"US", "GB", "CA"}

    def __init__(self, config):
        self.config = config
        self.raw_dir = config["data"]["raw_dir"]
        self.processed_dir = config["data"]["processed_dir"]
        self.clean_df = None
        self.labeled_df = None

    def find_csv_files(self):
        """Find CSV files in raw directory, excluding trending.csv."""
        csv_files = [f for f in os.listdir(self.raw_dir)
                     if f.endswith(".csv") and f != "trending.csv"]

        if not csv_files:
            print(f"[Data] No CSV files found in {self.raw_dir}.")
            try:
                import google.colab
                from google.colab import files as _cf
                uploaded = _cf.upload()
                for fname, content in uploaded.items():
                    with open(os.path.join(self.raw_dir, fname), "wb") as fh:
                        fh.write(content)
                    print(f"[Data] Saved: {fname}")
                csv_files = [f for f in os.listdir(self.raw_dir)
                             if f.endswith(".csv") and f != "trending.csv"]
            except ImportError:
                raise FileNotFoundError(
                    f"Place CSVs in {os.path.abspath(self.raw_dir)} and re-run."
                )

        detected = {f.replace("videos.csv", "").replace("videos", "") for f in csv_files}
        non_en = detected - self.ENGLISH_MARKETS
        if non_en:
            print(f"[Data] WARNING: non-English market files detected: {non_en}")
            print("[Data]          SigLIP text encoder is English-only — accuracy may degrade.")

        print(f"[Data] Found {len(csv_files)} CSV file(s): {csv_files}")
        return csv_files

    def load_and_clean(self):
        """Load all CSVs, combine, and clean."""
        csv_files = self.find_csv_files()

        print(f"[Data] Loading {len(csv_files)} CSV file(s)...")
        all_dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(os.path.join(self.raw_dir, csv_file),
                                 on_bad_lines="skip", low_memory=False, encoding="utf-8")
                if "video_id" not in df.columns and "id" in df.columns:
                    df = df.rename(columns={"id": "video_id"})
                all_dfs.append(df)
                print(f"[Data]   {csv_file}: {len(df):,} rows")
            except Exception as e:
                print(f"[Data]   ERROR loading {csv_file}: {e}")

        if not all_dfs:
            raise ValueError("[Data] No data loaded — verify CSV files are valid and non-empty.")

        trending_df = pd.concat(all_dfs, ignore_index=True)
        trending_df.to_csv(f"{self.raw_dir}/trending.csv", index=False)
        print(f"[Data] Combined total : {len(trending_df):,} rows")

        self.clean_df = self._clean_dataset(trending_df)
        self.clean_df.to_csv(f"{self.processed_dir}/clean_dataset.csv", index=False)
        print(f"[Data] After cleaning : {len(self.clean_df):,} rows  ->  saved clean_dataset.csv")
        return self

    def _clean_dataset(self, df):
        """Deduplicate, drop invalid rows, retain required columns + date."""
        if "video_id" in df.columns:
            df = df.dropna(subset=["video_id"])
            df = df[df["video_id"].astype(str).str.strip() != ""]
        if "views" in df.columns:
            df["views"] = pd.to_numeric(df["views"], errors="coerce")
            df = df.dropna(subset=["views"])
            df = df[df["views"] > 0]
        if "channel_id" not in df.columns:
            df["channel_id"] = (df["channel_title"].astype(str)
                                if "channel_title" in df.columns else "unknown")
        if "title" not in df.columns:
            df["title"] = "Untitled"
        else:
            df["title"] = df["title"].fillna("Untitled").astype(str)
            df = df[df["title"].str.strip() != ""]
        if "video_id" in df.columns:
            df = df.drop_duplicates(subset=["video_id"], keep="last")

        if "trending_date" in df.columns:
            date_col = "trending_date"
        elif "publishedAt" in df.columns:
            date_col = "publishedAt"
        else:
            df["trending_date"] = "1970-01-01"
            date_col = "trending_date"

        df["trending_date"] = pd.to_datetime(df[date_col], format="mixed", errors="coerce")

        if "category_id" in df.columns:
            df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce").fillna(0).astype(int)
        else:
            df["category_id"] = 0

        if "trending_date" in df.columns:
            df["hour_of_day"] = df["trending_date"].dt.hour.fillna(12).astype(int)
            df["day_of_week"] = df["trending_date"].dt.dayofweek.fillna(3).astype(int)
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        else:
            df["hour_of_day"] = 12
            df["day_of_week"] = 3
            df["is_weekend"] = 0

        keep = ["video_id", "title", "views", "channel_title", "channel_id",
                "trending_date", "category_id", "hour_of_day", "day_of_week", "is_weekend"]
        return df[[c for c in keep if c in df.columns]].reset_index(drop=True)
