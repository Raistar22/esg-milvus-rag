import re

SECTORS         = ["Automobile", "Finance", "Banking", "FMCG", "ALL"]
FINANCIAL_YEARS = ["2023-2024", "2024-2025"]
BRSR_PARAMETERS = (
    [f"S{i}" for i in range(1, 8)] +
    [f"G{i}" for i in range(1, 8)] +
    [f"E{i}" for i in range(1, 8)] +
    ["ALL"]
)

def detect_brsr_parameter(text):
    text_upper = text.upper()
    for i in range(1, 8):
        if f"PRINCIPLE {i}" in text_upper or f"P{i}" in text_upper:
            if i in [1,2,3]:   return f"E{i}"
            elif i in [4,5,6]: return f"S{i-3}"
            else:              return f"G{i-6}"
    t = text.lower()
    e = sum(1 for k in ["emission","carbon","energy","water","waste","climate","ghg"] if k in t)
    s = sum(1 for k in ["employee","health","safety","community","training","diversity"] if k in t)
    g = sum(1 for k in ["governance","board","compliance","audit","ethics","policy"] if k in t)
    mx = max(e, s, g)
    if mx == 0: return "ALL"
    if e == mx: return "E1"
    elif s == mx: return "S1"
    else: return "G1"

def detect_sector(text):
    t = text.lower()
    if any(k in t for k in ["automobile","vehicle","automotive","car","ev"]): return "Automobile"
    if any(k in t for k in ["bank","banking","nbfc","rbi"]):                  return "Banking"
    if any(k in t for k in ["finance","financial services","insurance"]):     return "Finance"
    if any(k in t for k in ["fmcg","consumer goods","food","beverage"]):      return "FMCG"
    return "ALL"

def detect_financial_year(text):
    if "2024-25" in text or "2024-2025" in text: return "2024-2025"
    if "2023-24" in text or "2023-2024" in text: return "2023-2024"
    return "2023-2024"

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())
