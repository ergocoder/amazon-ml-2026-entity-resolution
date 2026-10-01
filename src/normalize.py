"""
Text normalization for business names and addresses.

Country-agnostic on purpose: nothing here filters to US/India, so France
(test-only) flows through the same code. Word lists include a few common
French forms as generic abbreviations, not as a country switch.
"""
import re
import unicodedata
from multiprocessing import Pool

import pandas as pd

try:
    # anyascii (ISC license) transliterates any script to Latin letters,
    # e.g. Kannada/Telugu/Devanagari names -> rough English spelling.
    from anyascii import anyascii
except ImportError:  # fallback: only strips accents (é -> e)
    anyascii = None

# Legal-form words removed from names (they differ between sources).
LEGAL_WORDS = {
    "private", "pvt", "pvt.", "limited", "ltd", "llp", "llc", "inc", "incorporated",
    "corp", "corporation", "co", "company", "plc", "lp", "pllc", "pc",
    "com", "www", "net", "org",  # website-style names: foo.com
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "cie",  # French legal forms
}

# Address abbreviations -> one canonical form.
ADDR_ABBREV = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "pkwy": "parkway", "hwy": "highway", "pl": "place", "sq": "square", "cir": "circle",
    "ter": "terrace", "trl": "trail", "fwy": "freeway", "expy": "expressway",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "nr": "near", "opp": "opposite", "bldg": "building", "flr": "floor", "fl": "floor",
    "ste": "suite", "apt": "apartment", "appartments": "apartments",
    "bengaluru": "bangalore", "calicut": "kozhikode",
}

# Filler words that carry no identity (dropped from addresses).
ADDR_FILLER = {"unit", "no", "number", "suite", "apartment", "flat", "shop", "plot",
               "h", "hno", "house", "k", "kh", "null", "the", "of"}

# Full state names -> short code, so "Alabama" and "AL" become the same token.
STATE_ALIASES = {
    # US
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
    # India (single-word names; multi-word ones handled in MULTI_WORD)
    "maharashtra": "mh", "karnataka": "ka", "telangana": "tg", "ts": "tg", "gujarat": "gj",
    "kerala": "kl", "rajasthan": "rj", "delhi": "dl", "haryana": "hr", "punjab": "pb",
    "bihar": "br", "odisha": "od", "orissa": "od", "assam": "as", "goa": "goa",
    "jharkhand": "jh", "uttarakhand": "uk", "chhattisgarh": "cg",
}
MULTI_WORD = [
    (r"\bnew york\b", "ny"), (r"\bnew jersey\b", "nj"), (r"\bnew mexico\b", "nm"),
    (r"\bnew hampshire\b", "nh"), (r"\bnorth carolina\b", "nc"), (r"\bsouth carolina\b", "sc"),
    (r"\bnorth dakota\b", "nd"), (r"\bsouth dakota\b", "sd"), (r"\brhode island\b", "ri"),
    (r"\bwest virginia\b", "wv"), (r"\bandhra pradesh\b", "ap"), (r"\buttar pradesh\b", "up"),
    (r"\bmadhya pradesh\b", "mp"), (r"\btamil nadu\b", "tn"), (r"\bwest bengal\b", "wb"),
    (r"\bhimachal pradesh\b", "hp"), (r"\bnew delhi\b", "delhi"),
]

_ORDINAL = re.compile(r"^\d+(st|nd|rd|th|er|e|eme)$")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s"})
_ACRONYM = re.compile(r"\b(?:[a-z]\.){2,}(?!\w)")          # l.l.p. -> llp
_AKA = re.compile(r"\b[fad]\s*/?\s*[kb]\s*/?\s*a\b")        # f/k/a, d/b/a, aka
_DIGIT_RUN = re.compile(r"\d+")


def _to_ascii(s: str) -> str:
    """Convert any script (Hindi, Kannada, accented French...) to plain Latin letters.
    Uses anyascii if installed, otherwise only strips accents. Returns the new string."""
    if anyascii is not None:
        return anyascii(s)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def _fix_leet(tok: str) -> str:
    """'c0mpany' -> 'company', but leave '2nd', 'e18', '109th' alone."""
    n_digit = sum(ch.isdigit() for ch in tok)
    n_alpha = sum(ch.isalpha() for ch in tok)
    if n_digit and n_alpha >= 3 and n_alpha > n_digit and not _ORDINAL.match(tok):
        return tok.translate(_LEET)
    return tok


def _base_tokens(s: str) -> list:
    """Shared cleanup: transliterate, lowercase, unify punctuation, split to tokens."""
    if not s:
        return []
    s = _to_ascii(s).lower()
    s = _AKA.sub(" ", s)
    s = _ACRONYM.sub(lambda m: m.group().replace(".", ""), s)
    s = s.replace("&", " and ")
    s = re.sub(r"(?<=[a-z])[-/](?=[a-z])", " ", s)   # su-security -> su security
    s = re.sub(r"[^a-z0-9/\- ]", " ", s)               # drop other punctuation
    s = re.sub(r"(?<![0-9])[-/](?![0-9])", " ", s)     # keep - and / only next to digits
    for pat, rep in MULTI_WORD:
        s = re.sub(pat, rep, s)
    toks = [t.strip("-/") for t in s.split()]
    # "004303" -> "4303" (leading zeros are formatting noise)
    toks = [(t.lstrip("0") or "0") if t.isdigit() else t for t in toks]
    return [_fix_leet(t) for t in toks if t]


def normalize_name(s: str) -> str:
    """Clean a raw business name: basic cleanup, then drop legal words (pvt, llc, inc...)
    and "and" unless nothing else is left. Returns space-separated tokens."""
    toks = _base_tokens(s)
    core = [t for t in toks if t not in LEGAL_WORDS and t != "and"]
    return " ".join(core if core else toks)


def normalize_address(s: str) -> str:
    """Clean a raw address: basic cleanup, unify abbreviations (rd -> road) and state
    names (Alabama -> al), drop filler words (unit, flat...). Returns space-separated tokens."""
    out = []
    for t in _base_tokens(s):
        t = ADDR_ABBREV.get(t, t)
        t = STATE_ALIASES.get(t, t)
        if t not in ADDR_FILLER:
            out.append(t)
    return " ".join(out)


def number_parts(addr_norm: str) -> list:
    """Digit runs inside compound numbers: '2702/1313' -> ['2702', '1313']."""
    parts = []
    for tok in addr_norm.split():
        if not tok.isdigit():
            parts.extend(r for r in _DIGIT_RUN.findall(tok) if len(r) >= 2)
    return parts


def _normalize_row(pair):
    """Normalize one (raw name, raw address) pair. Returns (name_n, addr_n, block_text),
    where block_text = name + address + number parts, used for blocking."""
    name, addr = pair
    n, a = normalize_name(name), normalize_address(addr)
    block_text = " ".join([n, a] + number_parts(a))
    return n, a, block_text


def add_normalized(df: pd.DataFrame, n_jobs: int = 4, chunksize: int = 20000) -> pd.DataFrame:
    """Adds name_n, addr_n, block_text columns. Uses several CPU cores."""
    rows = list(zip(df["business_name"], df["business_address"]))
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            res = pool.map(_normalize_row, rows, chunksize=chunksize)
    else:
        res = [_normalize_row(r) for r in rows]
    df = df.copy()
    df["name_n"], df["addr_n"], df["block_text"] = zip(*res) if res else ([], [], [])
    return df
