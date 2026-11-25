"""
text_features.py — turn fighter profiles into numbers (the TEXT modality).

Per fighter, three kinds of text features:
  1. Keyword flags: martial arts background named in the profile
     (wrestling, BJJ, judo, sambo, boxing, muay thai, kickboxing, karate, taekwondo)
  2. TF-IDF + SVD: a compact summary of all the words (16 numbers)
  3. MiniLM sentence embeddings + PCA (16 numbers), if fighter_embeddings.csv exists
     (made on a machine with internet by embed_profiles.py)

The leak filter from scrape_profiles.py is re-applied on load, so the newest
filter rules hold even if the profiles were scraped with an older version.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scrape_profiles import remove_leaky_sentences  # noqa: E402

KEYWORDS = {
    "wrestling": r"wrestl",
    "bjj": r"jiu.?jitsu|\bbjj\b|grappl",
    "judo": r"\bjudo",
    "sambo": r"sambo",
    "boxing": r"\bbox(?:ing|er)\b|golden gloves",
    "muay_thai": r"muay thai",
    "kickboxing": r"kickbox",
    "karate": r"karate",
    "taekwondo": r"taekwondo|tae kwon do",
    "striker": r"fighting style: (?:striker|boxer|muay thai|kickbox|karate)",
    "grappler": r"fighting style: (?:grappler|wrestler|wrestling|jiu|bjj|judo|sambo)",
}
N_SVD = 16
N_EMB = 16


def load_profiles(path="data/fighter_profiles.csv"):
    p = pd.read_csv(path)
    wiki = p["wiki_text"].fillna("")
    wiki = wiki.where(~wiki.str.contains("may refer to", case=False), "")  # disambiguation pages
    p["wiki_text"] = wiki.map(remove_leaky_sentences)
    p["ufc_text"] = p["ufc_text"].fillna("")
    p["clean_text"] = (p["ufc_text"] + " " + p["wiki_text"]).str.strip()
    p["has_profile"] = (p["clean_text"] != "").astype(float)
    p.loc[p["clean_text"] == "", "clean_text"] = "No profile available."
    return p.drop_duplicates("fighter_name").set_index("fighter_name")


def fighter_text_features(p, emb_path="data/fighter_embeddings.csv"):
    """One row of text features per fighter."""
    low = p["clean_text"].str.lower()
    F = pd.DataFrame(index=p.index)
    F["has_profile"] = p["has_profile"]
    F["text_words"] = p["clean_text"].str.split().str.len() * p["has_profile"]
    for k, pat in KEYWORDS.items():
        F[f"kw_{k}"] = low.str.contains(pat, regex=True).astype(float)

    # TF-IDF + SVD (unsupervised: never sees fight outcomes)
    tf = TfidfVectorizer(min_df=3, max_df=0.5, ngram_range=(1, 2), stop_words="english")
    M = tf.fit_transform(p["clean_text"])
    k = min(N_SVD, M.shape[1] - 1, M.shape[0] - 1)
    if k >= 1:
        Z = TruncatedSVD(n_components=k, random_state=0).fit_transform(M)
        for i in range(k):
            F[f"tfidf_{i}"] = Z[:, i]

    # MiniLM embeddings, if available
    if os.path.exists(emb_path):
        E = pd.read_csv(emb_path).drop_duplicates("fighter_name").set_index("fighter_name")
        E = E.reindex(p.index)
        ok = E.notna().all(axis=1) & (p["has_profile"] == 1)
        k = min(N_EMB, int(ok.sum()) - 1)
        if k >= 1:
            Z = PCA(n_components=k, random_state=0).fit_transform(E[ok].values)
            for i in range(k):
                F[f"emb_{i}"] = np.nan
                F.loc[ok, f"emb_{i}"] = Z[:, i]
    return F


def fight_text_features(fights, F):
    """Map per-fighter text features onto each fight (red and blue corner)."""
    out = pd.DataFrame(index=fights.index)
    for corner in ("R", "B"):
        block = F.reindex(fights[f"{corner}_fighter"].values)
        block.index = fights.index
        block = block.fillna({"has_profile": 0.0})
        block.columns = [f"{corner}_txt_{c}" for c in block.columns]
        out = out.join(block)
    # matchup: who brings which background
    for k in KEYWORDS:
        out[f"txt_any_{k}"] = out[[f"R_txt_kw_{k}", f"B_txt_kw_{k}"]].max(axis=1)
    out["txt_both_profiles"] = out["R_txt_has_profile"] * out["B_txt_has_profile"]
    return out.astype(float)


def coverage_report(fights, F):
    has = F["has_profile"].reindex(pd.concat([fights.R_fighter, fights.B_fighter]).unique()).fillna(0)
    r = F["has_profile"].reindex(fights.R_fighter).fillna(0).values
    b = F["has_profile"].reindex(fights.B_fighter).fillna(0).values
    return {
        "fighters": len(has), "fighters_with_text": int(has.sum()),
        "fights": len(fights), "fights_both_have_text": int((r * b).sum()),
    }


class TextEncoder:
    """Saveable version of fighter_text_features (keywords + TF-IDF/SVD).

    fit() learns the TF-IDF vocabulary and SVD on the training profiles;
    transform() applies the same encoding to any profiles, including fighters
    the model has never seen. Used by predict.py so new uploads are encoded
    exactly like the training data. (MiniLM embeddings are not included.)
    """

    def fit(self, p):
        self.tf = TfidfVectorizer(min_df=3, max_df=0.5, ngram_range=(1, 2), stop_words="english")
        M = self.tf.fit_transform(p["clean_text"])
        self.k = min(N_SVD, M.shape[1] - 1, M.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=self.k, random_state=0).fit(M)
        return self

    def transform(self, p):
        low = p["clean_text"].str.lower()
        F = pd.DataFrame(index=p.index)
        F["has_profile"] = p["has_profile"]
        F["text_words"] = p["clean_text"].str.split().str.len() * p["has_profile"]
        for k, pat in KEYWORDS.items():
            F[f"kw_{k}"] = low.str.contains(pat, regex=True).astype(float)
        Z = self.svd.transform(self.tf.transform(p["clean_text"]))
        for i in range(self.k):
            F[f"tfidf_{i}"] = Z[:, i]
        return F


def clean_profiles(p):
    """Apply load_profiles' cleaning to an in-memory profiles table."""
    p = p.copy()
    wiki = p["wiki_text"].fillna("")
    wiki = wiki.where(~wiki.str.contains("may refer to", case=False), "")
    p["wiki_text"] = wiki.map(remove_leaky_sentences)
    p["ufc_text"] = p["ufc_text"].fillna("")
    p["clean_text"] = (p["ufc_text"] + " " + p["wiki_text"]).str.strip()
    p["has_profile"] = (p["clean_text"] != "").astype(float)
    p.loc[p["clean_text"] == "", "clean_text"] = "No profile available."
    return p.drop_duplicates("fighter_name").set_index("fighter_name")
