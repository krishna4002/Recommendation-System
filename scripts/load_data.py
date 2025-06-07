# scripts/load_data.py
import pandas as pd
def load_books(path="data/books.csv", n=None):
    df = pd.read_csv(path)
    df = df.dropna(subset=["title", "authors"])
    df["description"] = df["title"] + " by " + df["authors"]  # synthetic description
    return df if n is None else df.head(n)