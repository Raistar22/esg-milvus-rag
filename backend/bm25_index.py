from rank_bm25 import BM25Okapi
from backend.metadata import tokenize

_bm25 = None
_docs = None

def build_bm25(docs):
    global _bm25, _docs
    _docs  = docs
    _bm25  = BM25Okapi([tokenize(d.page_content) for d in docs])
    return _bm25

def get_bm25():
    return _bm25, _docs
