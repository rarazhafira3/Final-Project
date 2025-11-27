# src/preprocess.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(s):
    if pd.isna(s): return ""
    s = str(s)
    s = re.sub(r"http\S+", "", s)
    s = re.sub(r"<.*?>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def prepare(path_in="data/raw/idn-turnbackhoax-2025.csv",
            out_train="data/processed/train.csv",
            out_val="data/processed/val.csv",
            out_test="data/processed/test.csv",
            test_size=0.15, val_size=0.15, random_state=42):
    df = pd.read_csv(path_in)
    # adjust columns if different; try to detect 'title' and 'content'
    title_col = None
    content_col = None
    for c in df.columns:
        if 'title' in c.lower():
            title_col = c
        if 'content' in c.lower() or 'body' in c.lower() or 'isi' in c.lower():
            content_col = c
    df['title'] = df[title_col] if title_col else ""
    df['content'] = df[content_col] if content_col else ""
    df['text'] = (df['title'].fillna('') + " . " + df['content'].fillna('')).apply(clean_text)
    # map label options to hoax / non_hoax
    label_col = None
    for c in df.columns:
        if 'label' in c.lower() or 'kategori' in c.lower() or 'result' in c.lower():
            label_col = c
    if not label_col:
        raise ValueError("Label column not found. Check dataset columns: " + ",".join(df.columns))
    df['label'] = df[label_col].astype(str).str.lower().replace({
        'hoax':'hoax','fake':'hoax','not hoax':'non_hoax','not_hoax':'non_hoax','fakta':'non_hoax'
    })
    df = df[df['label'].isin(['hoax','non_hoax'])]
    df = df[['text','label']].dropna()
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])
    train, val = train_test_split(train, test_size=val_size/(1-test_size), random_state=random_state, stratify=train['label'])
    train.to_csv(out_train, index=False)
    val.to_csv(out_val, index=False)
    test.to_csv(out_test, index=False)
    print("Saved:", out_train, out_val, out_test)

if __name__ == "__main__":
    prepare()
