import pandas as pd
import re


def normalize_text(text):

    # handle empty values
    if pd.isna(text):
        return ""

    # convert to string
    text = str(text)

    # uppercase everything
    text = text.upper()

    # remove leading/trailing spaces
    text = text.strip()

    # replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)

    # standardize separators (collapse runs of - or / to a single slash)
    text = re.sub(r"[-/]+", "/", text)

    return text