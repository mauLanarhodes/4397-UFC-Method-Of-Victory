"""
scrape_profiles.py — build the TEXT modality for the UFC method-of-victory project.

For every fighter in the Kaggle "ufcdata" dataset (rajeevw/ufcdata), collect:
  1. Wikipedia: intro + background / early-life / fighting-style sections only
  2. UFC.com athlete page: "Fighting style", "Trains at", "Hometown" fields

Sentences that describe fight RESULTS (records, titles, wins/losses) are removed
to limit data leakage, because these pages are written today, not before each fight.

Output: fighter_profiles.csv
  fighter_name, wiki_title, wiki_text, ufc_url, ufc_text, text,
  has_wiki, has_ufc, has_profile

Usage:
  pip install requests beautifulsoup4 pandas
  python scrape_profiles.py --data data.csv --since 2010 --extra meta.csv
      (fighters from 2010 on, the training window, plus the 40 personal-set fighters)
  python scrape_profiles.py --data data.csv --since 2010 --extra meta.csv --limit 20   # quick test
  python scrape_profiles.py --debug-ufc "Israel Adesanya"   # inspect one UFC.com page

Progress is cached in profile_cache.json, so if it stops you can just rerun it.
"""

import argparse
import json
import os
import re
import time
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    # Wikipedia asks for a descriptive User-Agent with contact info.
    "User-Agent": "UH-ML-ClassProject/1.0 (student project; contact: your-email@example.com)"
}
WIKI_API = "https://en.wikipedia.org/w/api.php"
UFC_URL = "https://www.ufc.com/athlete/{slug}"
CACHE_FILE = "profile_cache.json"
WIKI_DELAY = 0.1  # seconds between Wikipedia requests
UFC_DELAY = 0.4   # seconds between UFC.com requests; be polite

# Wikipedia sections worth keeping (matched case-insensitively against section titles)
KEEP_SECTIONS = re.compile(
    r"background|early life|fighting style|style|training|amateur|biography",
    re.I,
)
# UFC.com labels worth keeping
UFC_LABELS = ["Fighting style", "Trains at", "Hometown", "Place of Birth"]

# Sentences matching any of these get dropped (leakage filter)
LEAK_PATTERNS = re.compile(
    r"\b\d{1,2}\s*[-–]\s*\d{1,2}(\s*[-–]\s*\d{1,2})?\b"  # records like 22-1 or 10-2-1
    r"|champion|title|ranked|ranking|retire"  # not "belt": keep "BJJ black belt"
    r"|defeat|knock(ed)? out|submitted|submission win|won |win |wins |lost |loss"
    r"|record|undefeated|streak|hall of fame|fight of the night|performance of the night"
    # career-span / after-the-fact info (when a career ended, later promotions, death)
    r"|retir|former|from \d{4} (until|to) \d{4}|\([^)]*\d{4}\s*[–-][^)]*\d{4}\)|died|death|passed away|best known|stint|currently|competitor from"
    # personal-life noise that says nothing about fighting
    r"|married|wife|husband|daughter|\bson\b|children|girlfriend|arrest|accused|charged|altercation|stabb|killed"
    # life after fighting
    r"|\bis now\b|\bnow (a|an|works|runs|owns|hosts)\b|youtube|podcast|labrador|\bdogs?\b"
    # fight results and outcomes described directly
    r"|decision|\brounds?\b|\bufc \d|victor|\bbeat\b|\bt?ko\b|finish"
    # signings, releases, other promotions (usually later career)
    r"|released|signed|contract|bellator|\bpfl\b|one championship|rizin|strikeforce|\bwec\b"
    # life outside / after MMA
    r"|\bwwe\b|wrestlemania|comedy|\bactor\b|\bfilm|prison|sentenc|custody|parole|convicted"
    # anything dated 2005 or later could describe events after a training fight
    r"|\b(200[5-9]|20[1-2]\d)\b"
    # career summaries and reputation (written with hindsight)
    r"|of all time|greatest|\bbest\b|of the year|\branks?\b|most successful|controvers"
    r"|accolade|award|honou?r|regarded|considered|widely|legend|elite|top \d|#\d|\bno\. ?\d"
    r"|nicknamed|popular|famous|known as|highest|fastest|youngest|oldest|first (fighter|person|man|woman)"
    r"|suspen|\bfined?\b|commission|brawl|usada|doping|tested positive|banned",
    re.I,
)


# ----------------------------------------------------------------- helpers
def load_fighters(path, since=None):
    df = pd.read_csv(path)
    if since and "date" in df.columns:
        df = df[pd.to_datetime(df["date"], format="mixed").dt.year >= since]
    for a, b in [("R_fighter", "B_fighter"), ("r_fighter", "b_fighter")]:
        if a in df.columns and b in df.columns:
            names = pd.concat([df[a], df[b]]).dropna().astype(str).str.strip()
            return sorted(names.unique())
    for a, b in [("fighter_1", "fighter_2")]:  # personal test set meta.csv
        if a in df.columns and b in df.columns:
            return sorted(pd.concat([df[a], df[b]]).dropna().astype(str).str.strip().unique())
    if "fighter_name" in df.columns:  # raw_fighter_details.csv
        return sorted(df["fighter_name"].dropna().astype(str).str.strip().unique())
    raise ValueError("Couldn't find fighter name columns (R_fighter/B_fighter or fighter_name).")


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm(s):
    return re.sub(r"[^a-z ]", "", strip_accents(s).lower()).strip()


def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)


def remove_leaky_sentences(text):
    kept = [s for s in split_sentences(text) if s and not LEAK_PATTERNS.search(s)]
    return " ".join(kept).strip()


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


def get(url, **kw):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15, **kw)
            if r.status_code == 429:  # rate limited
                time.sleep(5 * (attempt + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(2)
    return None


# ----------------------------------------------------------------- Wikipedia
def wiki_find_title(name):
    """Search Wikipedia and return the page title for this fighter, or None."""
    r = get(WIKI_API, params={
        "action": "query", "list": "search", "format": "json", "srlimit": 5,
        "srsearch": f'"{name}" mixed martial artist',
    })
    if not r or r.status_code != 200:
        return None
    target = set(norm(name).split())
    for hit in r.json().get("query", {}).get("search", []):
        title = hit["title"]
        base = re.sub(r"\(.*?\)", "", title)  # drop "(fighter)" etc.
        snippet = hit.get("snippet", "").lower()
        if target and target <= set(norm(base).split()) and (
            "martial" in snippet or "fighter" in snippet or "ufc" in snippet
            or "(fighter)" in title.lower()
        ):
            return title
    return None


def parse_wiki_sections(plain):
    """Keep the intro + whitelisted sections from a plain-text extract with == headers ==."""
    parts = re.split(r"\n(={2,})\s*(.*?)\s*\1\n", "\n" + plain)
    # parts = [intro, '==', title1, body1, '==', title2, body2, ...]
    kept = [parts[0]]
    for i in range(1, len(parts) - 2, 3):
        title, body = parts[i + 1], parts[i + 2]
        if KEEP_SECTIONS.search(title):
            kept.append(body)
    return re.sub(r"\s+", " ", " ".join(kept)).strip()


def wiki_profile(name):
    title = wiki_find_title(name)
    time.sleep(WIKI_DELAY)
    if not title:
        return None, ""
    r = get(WIKI_API, params={
        "action": "query", "prop": "extracts", "format": "json", "redirects": 1,
        "explaintext": 1, "exsectionformat": "wiki", "titles": title,
    })
    time.sleep(WIKI_DELAY)
    if not r or r.status_code != 200:
        return title, ""
    pages = r.json().get("query", {}).get("pages", {})
    plain = next(iter(pages.values()), {}).get("extract", "") or ""
    if "may refer to" in plain.lower():
        return None, ""  # disambiguation page, not the fighter's own page
    if "martial" not in plain.lower() and "fighter" not in plain.lower():
        return None, ""  # wrong person
    return title, remove_leaky_sentences(parse_wiki_sections(plain))


# ----------------------------------------------------------------- UFC.com
def ufc_slug(name):
    s = strip_accents(name).lower()
    s = re.sub(r"['.]", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def parse_ufc_fields(html):
    """Find label -> value pairs like 'Fighting style: Muay Thai' by label text,
    so it keeps working even if UFC.com renames its CSS classes."""
    soup = BeautifulSoup(html, "html.parser")
    fields = {}
    for label in UFC_LABELS:
        el = soup.find(string=re.compile(rf"^\s*{re.escape(label)}\s*:?\s*$", re.I))
        if not el:
            continue
        node = el.parent
        value = None
        # value is usually the next sibling element of the label element
        for sib in node.find_next_siblings():
            t = sib.get_text(" ", strip=True)
            if t:
                value = t
                break
        if not value and node.parent:  # or the next element within the parent block
            nxt = node.parent.find_next_sibling()
            if nxt:
                value = nxt.get_text(" ", strip=True)
        if value and len(value) < 120:
            fields[label] = value
    return fields


def ufc_profile(name):
    url = UFC_URL.format(slug=ufc_slug(name))
    r = get(url)
    time.sleep(UFC_DELAY)
    if not r or r.status_code != 200:
        return None, ""
    fields = parse_ufc_fields(r.text)
    if not fields:
        return url, ""
    text = ". ".join(f"{k}: {v}" for k, v in fields.items()) + "."
    return url, text


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv", help="Kaggle ufcdata CSV (data.csv)")
    ap.add_argument("--out", default="fighter_profiles.csv")
    ap.add_argument("--since", type=int, default=None, help="only fighters with a fight in or after this year")
    ap.add_argument("--extra", nargs="*", default=[], help="more CSVs of fighters to include, e.g. meta.csv")
    ap.add_argument("--limit", type=int, default=None, help="only do the first N fighters (testing)")
    ap.add_argument("--debug-ufc", metavar="NAME", help="print what the UFC.com parser finds for one fighter")
    args = ap.parse_args()

    if args.debug_ufc:
        url = UFC_URL.format(slug=ufc_slug(args.debug_ufc))
        r = get(url)
        print(url, r.status_code if r else "no response")
        if r is not None:
            with open("ufc_debug.html", "w", encoding="utf-8") as f:
                f.write(r.text)
            print("Saved raw page to ufc_debug.html")
            print("Parsed fields:", parse_ufc_fields(r.text))
        return

    fighters = load_fighters(args.data, args.since)
    for extra in args.extra:
        fighters = sorted(set(fighters) | set(load_fighters(extra)))
    if args.limit:
        fighters = fighters[: args.limit]
    print(f"{len(fighters)} fighters to process")

    cache = load_cache()
    for i, name in enumerate(fighters, 1):
        if name in cache:
            continue
        wiki_title, wiki_text = wiki_profile(name)
        ufc_url, ufc_text = ufc_profile(name)
        cache[name] = {
            "wiki_title": wiki_title, "wiki_text": wiki_text,
            "ufc_url": ufc_url, "ufc_text": ufc_text,
        }
        print(f"[{i}/{len(fighters)}] {name:30s} wiki={'Y' if wiki_text else '-'} ufc={'Y' if ufc_text else '-'}")
        if i % 25 == 0:
            save_cache(cache)
    save_cache(cache)

    rows = []
    for name in fighters:
        c = cache[name]
        combined = " ".join(t for t in [c["ufc_text"], c["wiki_text"]] if t).strip()
        rows.append({
            "fighter_name": name,
            "wiki_title": c["wiki_title"], "wiki_text": c["wiki_text"],
            "ufc_url": c["ufc_url"], "ufc_text": c["ufc_text"],
            "text": combined if combined else "No profile available.",
            "has_wiki": int(bool(c["wiki_text"])),
            "has_ufc": int(bool(c["ufc_text"])),
            "has_profile": int(bool(combined)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    # Coverage summary — put these numbers in the report / show the TA
    n = len(out)
    print("\n=== Coverage ===")
    print(f"Fighters:            {n}")
    print(f"Wikipedia text:      {out.has_wiki.sum()} ({out.has_wiki.mean():.0%})")
    print(f"UFC.com text:        {out.has_ufc.sum()} ({out.has_ufc.mean():.0%})")
    print(f"Any text:            {out.has_profile.sum()} ({out.has_profile.mean():.0%})")
    lens = out.loc[out.has_profile == 1, "text"].str.split().str.len()
    if len(lens):
        print(f"Avg words (w/ text): {lens.mean():.0f}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
