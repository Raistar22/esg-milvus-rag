import numpy as np
from backend.bm25_index    import get_bm25
from backend.milvus_store  import dense_search
from backend.embeddings    import get_embeddings
from backend.metadata      import tokenize

def build_filter(sector, financial_year, brsr_parameter):
    c = []
    if sector and sector != "ALL":         c.append(f'sector == "{sector}"')
    if financial_year:                     c.append(f'financial_year == "{financial_year}"')
    if brsr_parameter and brsr_parameter != "ALL": c.append(f'brsr_parameter == "{brsr_parameter}"')
    return " and ".join(c)

def reciprocal_rank_fusion(dense_hits, bm25_indices,
                            dense_weight=0.6, bm25_weight=0.4, k=60):
    scores = {}
    for rank, hit in enumerate(dense_hits):
        did = hit["id"]
        scores[did] = scores.get(did, 0) + dense_weight / (k + rank + 1)
    for rank, idx in enumerate(bm25_indices):
        scores[idx] = scores.get(idx, 0) + bm25_weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_search(question, sector="ALL", financial_year="2023-2024",
                  brsr_parameter="ALL", k=5,
                  dense_weight=0.6, bm25_weight=0.4):
    embeddings       = get_embeddings()
    bm25_index, docs = get_bm25()
    query_vector     = embeddings.embed_query(question)
    filter_expr      = build_filter(sector, financial_year, brsr_parameter)
    dense_hits       = dense_search(query_vector, filter_expr, k * 3)

    bm25_scores = bm25_index.get_scores(tokenize(question))
    if filter_expr:
        for idx, doc in enumerate(docs):
            m = doc.metadata
            if sector != "ALL" and m.get("sector") != sector:           bm25_scores[idx] = 0.0
            if m.get("financial_year") != financial_year:                bm25_scores[idx] = 0.0
            if brsr_parameter != "ALL" and m.get("brsr_parameter") != brsr_parameter: bm25_scores[idx] = 0.0

    bm25_ranked = np.argsort(bm25_scores)[::-1][:k*3].tolist()
    fused       = reciprocal_rank_fusion(dense_hits, bm25_ranked, dense_weight, bm25_weight)
    top_k_ids   = [did for did, _ in fused[:k]]
    fused_dict  = dict(fused)
    dense_map   = {hit["id"]: hit for hit in dense_hits}

    chunks = []
    for doc_id in top_k_ids:
        if doc_id in dense_map:
            e = dense_map[doc_id]["entity"]
            chunks.append({**e, "source":"dense", "doc_id":doc_id, "fused_score":fused_dict[doc_id]})
        else:
            doc = docs[doc_id]
            chunks.append({
                "text":           doc.page_content[:2000],
                "sector":         doc.metadata.get("sector","ALL"),
                "financial_year": doc.metadata.get("financial_year","2023-2024"),
                "brsr_parameter": doc.metadata.get("brsr_parameter","ALL"),
                "esg_category":   doc.metadata.get("esg_category","ALL"),
                "page_number":    int(doc.metadata.get("page_number",0)),
                "source":         "bm25", "doc_id":doc_id, "fused_score":fused_dict[doc_id]
            })
    return chunks, filter_expr
