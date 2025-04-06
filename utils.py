import pandas as pd
from sklearn.utils import resample

# Cleaning and Loading Data
def filtering_df(df):
    df = pd.read_csv(df)
    df['name'] = df['name'].str.strip()
    df_cleaned = df.drop(columns=['avatar', 'is_real'])

    # Celebs with >= 5 tweets in original data
    df_filtered = df_cleaned.groupby('name').filter(lambda x: len(x) >= 5)

    # Resample to 8 tweets per celeb (duplicate if needed)
    balanced_dfs = []

    for name, group in df_filtered.groupby('name'):
        if len(group) >= 8:
            sampled = group.sample(n=8, random_state=42)
        else:
            sampled = resample(group, replace=True, n_samples=8, random_state=42)
        balanced_dfs.append(sampled)

    # Combine into one final DataFrame
    df_balanced = pd.concat(balanced_dfs).reset_index(drop=True)

    # Save to CSV
    df_balanced.to_csv("top_celebs.csv", index=False)
filtering_df('data/tweets.csv')

import base64

def get_image_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
