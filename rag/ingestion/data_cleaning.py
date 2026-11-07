"""Text cleaning, flags and category rules for the RAG pool (pure functions, easy to test)."""
import re, unicodedata

_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
DEVANAGARI = re.compile("[\u0900-\u097F]")
FIGURES = re.compile(r"(?:\d+(?:\.\d+)?\s?%|rs\.?\s?\d|\u20b9\s?\d|\blakh|\bcrore|per year|per annum|interest rate|\bfee\b|\bfees\b|charges?|\blimit)", re.I)
CONTACT = re.compile(r"(https?://|www\.|\b\d{10}\b|\b\d{4}[\s-]\d{3}[\s-]\d{4}\b|[\w.]+@[\w.]+\.\w+)", re.I)
AMBIGUOUS_Q = re.compile(r"\b(this|the|my|your)\s+(card|plan|policy|account|facility|loan|product|scheme|programme|program)\b", re.I)

# The workbook still holds 81 broken rupee signs: UTF-8 bytes of the rupee sign were read as Windows-1252 ("\u00e2\u201a\u00b9")
# and the superscript one was then flattened to a plain "1". So "\u00e2\u201a" is ALWAYS followed by that extra "1"
# (verified on all 81 cases). Without repair "Rs 5 lakh" would read "15 lakh".
_BROKEN_RUPEE = "\u00e2\u201a1"

def clean_text(s) -> str:
    """Repair broken rupee signs, Unicode-normalise, remove invisible characters, collapse whitespace."""
    if s is None: return ""
    s = str(s).replace(_BROKEN_RUPEE, "\u20b9")
    s = unicodedata.normalize("NFC", s).replace("\xa0", " ")
    s = _ZERO_WIDTH.sub("", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()

def clean_title(s) -> str:
    """clean_text plus removal of the synthetic ' - query 12' variant tag."""
    return re.sub(r"\s*[-\u2014]\s*query\s*\d+\s*$", "", clean_text(s), flags=re.I).strip()

def norm_key(s) -> str:
    """Lower-case, no punctuation: used only to compare texts for exact-meaning duplicates."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).split())

def fact_check_reasons(question, answer) -> list:
    r = []
    if FIGURES.search(answer): r.append("figure_fee_limit_or_rate")
    if CONTACT.search(answer): r.append("contact_or_url")
    if re.search(r"hdfc", question + " " + answer, re.I): r.append("hdfc_specific_claim")
    return r

def is_product_ambiguous(question) -> bool:
    return bool(AMBIGUOUS_Q.search(question))

CATEGORY_RULES = [
    ("Forex & Travel Cards", r"forex|multicurrency|hajj|umrah|travel card|foreign exchange|remittance"),
    ("Credit Cards", r"credit card|credit limit|billing cycle|reward point"),
    ("Debit & ATM", r"debit card|\batm\b"),
    ("Cards (general)", r"\bcard\b"),
    ("Loans", r"\bloan|\bemi\b|mortgage|larr|collateral|overdraft"),
    ("Digital Banking", r"net ?banking|mobile banking|\bupi\b|\bimps\b|\bneft\b|\brtgs\b|\bapp\b|\bpin\b|pattern lock|\bpassword|\bsms\b|\btab\b"),
    ("Insurance & Investments", r"insurance|\bplan\b|policy|fund value|premium|mutual fund|annuity|loyalty addition"),
    ("Accounts & Deposits", r"savings|current account|sweep|super saver|deposit|\bfd\b|\brd\b|account"),
]
def assign_category(question, answer):
    for name, pat in CATEGORY_RULES:
        if re.search(pat, question, re.I): return name
    for name, pat in CATEGORY_RULES:
        if re.search(pat, answer, re.I): return name
    return "General Banking"
