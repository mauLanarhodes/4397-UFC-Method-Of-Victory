"""
scrape_ufcstats.py — collect raw UFCStats data for the personal test set.

For each of the 20 personal-set fights (UFC 317 and UFC 320, 2025) this script:
  1. finds the fight on the event page,
  2. opens both fighters' pages (height, reach, weight, stance, date of birth),
  3. scrapes EVERY UFC fight on each fighter's record, in the same raw format as the
     Kaggle file raw_total_fight_data.csv (so ufc_features.py can compute pre-fight
     stats exactly like the training data).

Page parsing follows the original dataset's scraper
(github.com/WarrierRajeev/UFC-Predictions, src/createdata/scrape_*.py).

Outputs (in personal_test_set/):
  raw_fights_2025.csv      every fight on the 40 fighters' records (semicolon-separated)
  fighter_details_2025.csv height, weight, reach, stance, DOB for the 40 fighters
  target_fights.csv        sample_id -> red/blue corner, fight URL, event, date

Usage:
  pip install requests beautifulsoup4 pandas
  python scrape_ufcstats.py
  python scrape_ufcstats.py --debug      # save the first fight page as ufcstats_debug.html
  python scrape_ufcstats.py --events <UFC 317 event link> <UFC 320 event link>

Progress is cached in ufcstats_cache.json, so rerunning continues where it stopped.
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

BASE = "http://ufcstats.com"
EVENTS_URL = BASE + "/statistics/events/completed?page=all"
EVENTS = {"UFC 317": "2025-06-28", "UFC 320": "2025-10-04"}
META = "personal_test_set/meta.csv"
OUT = "personal_test_set/"
CACHE = "ufcstats_cache.json"
DELAY = 0.4
HEADER = ["R_fighter", "B_fighter", "R_KD", "B_KD", "R_SIG_STR.", "B_SIG_STR.", "R_SIG_STR_pct",
          "B_SIG_STR_pct", "R_TOTAL_STR.", "B_TOTAL_STR.", "R_TD", "B_TD", "R_TD_pct", "B_TD_pct",
          "R_SUB_ATT", "B_SUB_ATT", "R_REV", "B_REV", "R_CTRL", "B_CTRL", "R_HEAD", "B_HEAD",
          "R_BODY", "B_BODY", "R_LEG", "B_LEG", "R_DISTANCE", "B_DISTANCE", "R_CLINCH", "B_CLINCH",
          "R_GROUND", "B_GROUND", "win_by", "last_round", "last_round_time", "Format", "Referee",
          "date", "location", "Fight_type", "Winner"]
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "UH-ML-ClassProject/1.0 (student project)"


# ----------------------------------------------------------------- helpers
def norm(s):
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", "", s.lower().replace("-", " ")).split()


def same_person(a, b):
    """Loose name match: same last name and at least one more shared name part."""
    ta, tb = norm(a), norm(b)
    ta = [t for t in ta if t not in ("jr", "sr", "ii", "iii")]
    tb = [t for t in tb if t not in ("jr", "sr", "ii", "iii")]
    if not (ta and tb) or ta[-1] != tb[-1]:
        return False
    if len(set(ta) & set(tb)) >= min(2, len(ta), len(tb)):
        return True
    # short forms of a first name, e.g. "Abus" for "Abusupiyan", "Joe" for "Joseph"
    nick = {"joe": "joseph", "mike": "michael", "alex": "alexander", "tony": "anthony"}
    a, b = nick.get(ta[0], ta[0]), nick.get(tb[0], tb[0])
    return a.startswith(b) or b.startswith(a)


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {"pages": {}}


def soup(url, cache):
    if url not in cache["pages"]:
        for attempt in range(3):
            try:
                r = SESSION.get(url, timeout=20)
                r.raise_for_status()
                cache["pages"][url] = r.text
                break
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                time.sleep(3)
        time.sleep(DELAY)
    return BeautifulSoup(cache["pages"][url], "html.parser")


def save_cache(cache):
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def clean(t):
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------- page parsers
def event_links(cache, manual=None):
    """Find the event pages. Searches every link on the events list, not just one CSS class."""
    found = {}
    if manual:
        for url in manual:
            title = clean(soup(url, cache).select_one("h2, title").text) if url else ""
            for key in EVENTS:
                if key in title:
                    found[key] = url
        return found
    s = soup(EVENTS_URL, cache)
    for a in s.find_all("a", href=True):
        name = clean(a.text)
        for key in EVENTS:
            if re.match(rf"^{re.escape(key)}(\D|$)", name) and "event-details" in a["href"]:
                found[key] = a["href"]
    if len(found) < len(EVENTS):
        page = cache["pages"].pop(EVENTS_URL, "")     # don't keep a bad copy in the cache
        title = s.title.text.strip() if s.title else "(no title)"
        names = [clean(a.text) for a in s.find_all("a", href=True) if "event-details" in a["href"]][:5]
        print("  Events page diagnostics:")
        print(f"    page size: {len(page)} characters, title: {title!r}")
        print(f"    event links found: {len(names)}; first few: {names}")
        print("    If the title looks like a block/captcha page, open the events page in a browser,")
        print("    copy the UFC 317 and UFC 320 event links, and run:")
        print("      python scrape_ufcstats.py --events <UFC 317 link> <UFC 320 link>")
    return found


def event_info(s):
    """Date and location, as in the original (first two list items on the event page)."""
    items = [clean(li.text) for li in s.find_all("li", {"class": "b-list__box-list-item"})]
    date = next((i.replace("Date:", "").strip() for i in items if i.startswith("Date:")), "")
    loc = next((i.replace("Location:", "").strip() for i in items if i.startswith("Location:")), "")
    return date, loc


def fight_links_on_event(s):
    return [tr.get("data-link") for tr in s.find_all("tr", {"class": "b-fight-details__table-row"})
            if tr.get("data-link")]


def parse_fight(s):
    """One fight page -> dict with the raw columns (except date/location)."""
    persons = s.find_all("div", {"class": "b-fight-details__person"})
    names, links, winner = [], [], ""
    for p in persons:
        a = p.find("a", {"class": "b-fight-details__person-link"}) or p.find("a")
        h3 = p.find("h3", {"class": "b-fight-details__person-name"})
        nm = clean(h3.text) if h3 else clean(a.text)
        names.append(nm)
        links.append(a["href"] if a and a.get("href") else "")
        status = p.find("i", {"class": "b-fight-details__person-status"})
        if status and "b-fight-details__person-status_style_green" in status.get("class", []):
            winner = nm

    tables = s.find_all("tbody")
    if len(tables) < 3:
        raise ValueError("fight page has no stats tables (fight may lack stats)")

    def row_pairs(table):
        cols = []
        for td in table.find("tr").find_all("td"):
            ps = [clean(p.text) for p in td.find_all("p")]
            cols.append(ps if len(ps) >= 2 else [clean(td.text), ""])
        return cols

    totals = row_pairs(tables[0])       # Fighter, KD, Sig.str, Sig.str%, Total str, Td, Td%, Sub, Rev, Ctrl
    sig = row_pairs(tables[2])          # Fighter, Sig.str, Sig.str%, Head, Body, Leg, Distance, Clinch, Ground
    t = {}
    keys_tot = ["fighter", "KD", "SIG_STR.", "SIG_STR_pct", "TOTAL_STR.", "TD", "TD_pct", "SUB_ATT", "REV", "CTRL"]
    for k, (r, b) in zip(keys_tot, totals):
        t["R_" + k], t["B_" + k] = r, b
    for k, (r, b) in zip(["fighter", "SIG", "SIGP", "HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"], sig):
        if k in ("HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"):
            t["R_" + k], t["B_" + k] = r, b

    details = {}
    for p in s.find_all("p", {"class": "b-fight-details__text"}):
        for item in p.find_all("i", recursive=False) or [p]:
            txt = clean(item.text)
            for label in ("Method:", "Round:", "Time:", "Time format:", "Referee:"):
                if txt.startswith(label):
                    details[label] = txt[len(label):].strip()
    title = s.find("i", {"class": "b-fight-details__fight-title"})
    ev = s.select_one("h2.b-content__title a")
    row = {
        "R_fighter": names[0], "B_fighter": names[1],
        "win_by": details.get("Method:", ""), "last_round": details.get("Round:", ""),
        "last_round_time": details.get("Time:", ""), "Format": details.get("Time format:", ""),
        "Referee": details.get("Referee:", ""), "Fight_type": clean(title.text) if title else "",
        "Winner": winner,
        "_fighter_links": links, "_event_link": ev["href"] if ev else "",
    }
    for c in HEADER:
        if c not in row and c not in ("date", "location"):
            row[c] = t.get(c, "")
    return row


def fighter_page(s):
    """Fighter details (as in the original) + links to every fight on their record."""
    items = s.find_all("li", {"class": "b-list__box-list-item b-list__box-list-item_type_block"})
    det = {}
    for li in items:
        txt = clean(li.text)
        for label, key in (("Height:", "Height"), ("Weight:", "Weight"), ("Reach:", "Reach"),
                           ("STANCE:", "Stance"), ("DOB:", "DOB")):
            if txt.startswith(label):
                val = txt[len(label):].strip()
                det[key] = None if val in ("--", "") else val
    name = clean(s.find("span", {"class": "b-content__title-highlight"}).text)
    fights = [tr.get("data-link") for tr in s.find_all("tr", {"class": "b-fight-details__table-row"})
              if tr.get("data-link") and "fight-details" in tr.get("data-link")]
    return name, det, fights


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--events", nargs="+", help="event page links, if the events list can't be read")
    args = ap.parse_args()

    meta = pd.read_csv(META)
    cache = load_cache()

    print("Finding events...")
    evs = event_links(cache, args.events)
    missing = set(EVENTS) - set(evs)
    if missing:
        raise SystemExit(f"Could not find event(s) {missing} on {EVENTS_URL}")

    # 1) target fights
    targets, fighter_links = [], {}
    for key, url in evs.items():
        es = soup(url, cache)
        date, loc = event_info(es)
        for fl in fight_links_on_event(es):
            fs = soup(fl, cache)
            if args.debug:
                with open("ufcstats_debug.html", "w", encoding="utf-8") as f:
                    f.write(cache["pages"][fl])
                print("Saved ufcstats_debug.html")
                save_cache(cache)
                return
            try:
                fr = parse_fight(fs)
            except Exception as e:
                print("  skip fight (no stats):", fl, e)
                continue
            for _, m in meta[meta.date == EVENTS[key]].iterrows():
                pair = (m.fighter_1, m.fighter_2)
                rb = (fr["R_fighter"], fr["B_fighter"])
                if (same_person(pair[0], rb[0]) and same_person(pair[1], rb[1])) or \
                   (same_person(pair[0], rb[1]) and same_person(pair[1], rb[0])):
                    targets.append({"sample_id": m.sample_id, "R_fighter": rb[0], "B_fighter": rb[1],
                                    "fight_url": fl, "event": key, "date": date})
                    for nm, link in zip(rb, fr["_fighter_links"]):
                        fighter_links[nm] = link
        save_cache(cache)
    print(f"Matched {len(targets)} of {len(meta)} personal-set fights")
    unmatched = set(meta.sample_id) - {t["sample_id"] for t in targets}
    if unmatched:
        print("  NOT matched:", sorted(unmatched), "(check names in meta.csv)")

    # 2) every fight on each fighter's record
    details, fight_urls = [], set()
    for i, (nm, link) in enumerate(sorted(fighter_links.items()), 1):
        name, det, fights = fighter_page(soup(link, cache))
        details.append({"fighter_name": name, **{k: det.get(k) for k in ("Height", "Weight", "Reach", "Stance", "DOB")}})
        fight_urls.update(fights)
        print(f"[{i}/{len(fighter_links)}] {name}: {len(fights)} fights")
        save_cache(cache)

    rows = []
    for j, fl in enumerate(sorted(fight_urls), 1):
        try:
            fr = parse_fight(soup(fl, cache))
        except Exception as e:
            print("  skip (no stats):", fl)
            continue
        date, loc = event_info(soup(fr["_event_link"], cache)) if fr["_event_link"] else ("", "")
        fr["date"], fr["location"] = date, loc
        rows.append({c: fr[c] for c in HEADER})
        if j % 25 == 0:
            print(f"  fights scraped: {j}/{len(fight_urls)}")
            save_cache(cache)
    save_cache(cache)

    raw = pd.DataFrame(rows, columns=HEADER)
    raw["_d"] = pd.to_datetime(raw["date"], format="mixed", errors="coerce")
    raw = raw.sort_values("_d", ascending=False).drop(columns="_d")   # newest first, like Kaggle
    raw.to_csv(OUT + "raw_fights_2025.csv", sep=";", index=False)
    pd.DataFrame(details).to_csv(OUT + "fighter_details_2025.csv", index=False)
    pd.DataFrame(targets).sort_values("sample_id").to_csv(OUT + "target_fights.csv", index=False)
    print(f"\nSaved {len(raw)} fights, {len(details)} fighters, {len(targets)} target fights to {OUT}")


if __name__ == "__main__":
    main()
