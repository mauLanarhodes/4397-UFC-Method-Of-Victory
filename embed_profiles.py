"""
embed_profiles.py — make MiniLM sentence embeddings for every fighter profile.
Run this on a computer with internet (it downloads a ~90 MB model once).

  pip install sentence-transformers pandas
  python embed_profiles.py

Needs fighter_profiles.csv and scrape_profiles.py in the same folder.
Writes fighter_embeddings.csv (fighter_name + 384 numbers). Takes ~1–3 minutes on a laptop.
"""
import pandas as pd
from sentence_transformers import SentenceTransformer

from scrape_profiles import remove_leaky_sentences

p = pd.read_csv("fighter_profiles.csv")
wiki = p["wiki_text"].fillna("")
wiki = wiki.where(~wiki.str.contains("may refer to", case=False), "")  # disambiguation pages
wiki = wiki.map(remove_leaky_sentences)
text = (p["ufc_text"].fillna("") + " " + wiki).str.strip()
text = text.where(text != "", "No profile available.")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(text.tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)

out = pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
out.insert(0, "fighter_name", p["fighter_name"])
out.to_csv("fighter_embeddings.csv", index=False)
print(f"Saved {len(out)} embeddings -> fighter_embeddings.csv")
