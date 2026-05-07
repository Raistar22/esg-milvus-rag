from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
        )
    return _embeddings
