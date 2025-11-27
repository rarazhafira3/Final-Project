# src/preprocess.py
import pandas as pd
import re

def clean_text(s):
    if pd.isna(s): return ""
    s = str(s)
    s = re.sub(r"http\S+", "", s)              # remove urls
    s = re.sub(r"<.*?>", "", s)                # remove html tags
    s = re.sub(r"\s+", " ", s).strip()         # normalize whitespace
    return s.lower()

def prepare(path_in, out_train, out_val, out_test, test_size=0.15, val_size=0.15, random_state=42):
    df = pd.read_csv(path_in)
    # adapt column names if needed:
    # assume df has 'title' and 'content' and 'label'
    df['text'] = df.get('title', '').fillna('') + " . " + df.get('content', '').fillna('')
    df['text'] = df['text'].apply(clean_text)
    df['label'] = df['label'].str.lower().map({'hoax':'hoax','not hoax':'non_hoax','not_hoax':'non_hoax'}) 
    df = df.dropna(subset=['label','text'])
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])
    train, val = train_test_split(train, test_size=val_size/(1-test_size), random_state=random_state, stratify=train['label'])
    train[['text','label']].to_csv(out_train, index=False)
    val[['text','label']].to_csv(out_val, index=False)
    test[['text','label']].to_csv(out_test, index=False)
    print("Saved:", out_train, out_val, out_test)
