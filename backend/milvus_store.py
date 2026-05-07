from pymilvus import MilvusClient
from backend.embeddings import get_embeddings
from backend.metadata import detect_brsr_parameter, detect_sector, detect_financial_year
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

MILVUS_URI      = "./milvus_esg.db"
COLLECTION_NAME = "esg_reports"
_client         = None

def get_client():
    global _client
    if _client is None:
        _client = MilvusClient(uri=MILVUS_URI)
    return _client

def is_ingested():
    return get_client().has_collection(COLLECTION_NAME)

def ingest_pdf(pdf_path, progress_cb=None):
    embeddings = get_embeddings()
    client     = get_client()
    loader     = PyPDFLoader(pdf_path)
    raw        = loader.load()
    splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs       = splitter.split_documents(raw)

    for i, doc in enumerate(docs):
        t    = doc.page_content
        brsr = detect_brsr_parameter(t)
        doc.metadata.update({
            "sector":         detect_sector(t),
            "financial_year": detect_financial_year(t),
            "brsr_parameter": brsr,
            "esg_category":   ("Environmental" if brsr.startswith("E")
                               else "Social" if brsr.startswith("S")
                               else "Governance" if brsr.startswith("G")
                               else "ALL"),
            "page_number":    int(doc.metadata.get("page", i)),
            "chunk_id":       i,
        })

    texts   = [d.page_content for d in docs]
    vectors = embeddings.embed_documents(texts)
    if progress_cb: progress_cb(0.7, f"Embedded {len(vectors)} chunks")

    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=len(vectors[0]),
        metric_type="COSINE",
        auto_id=False
    )

    data = [{
        "id": i, "vector": vec,
        "text":           docs[i].page_content[:2000],
        "sector":         docs[i].metadata["sector"],
        "financial_year": docs[i].metadata["financial_year"],
        "brsr_parameter": docs[i].metadata["brsr_parameter"],
        "esg_category":   docs[i].metadata["esg_category"],
        "page_number":    docs[i].metadata["page_number"],
    } for i, vec in enumerate(vectors)]

    for start in range(0, len(data), 100):
        client.insert(COLLECTION_NAME, data[start:start+100])
        if progress_cb:
            progress_cb(0.7 + 0.3*(start/len(data)),
                        f"Inserted {min(start+100,len(data))}/{len(data)}")
    return len(docs), docs

def dense_search(query_vector, filter_expr, k):
    kwargs = dict(
        collection_name=COLLECTION_NAME,
        data=[query_vector], limit=k,
        output_fields=["text","sector","financial_year",
                       "brsr_parameter","esg_category","page_number"]
    )
    if filter_expr:
        kwargs["filter"] = filter_expr
    return get_client().search(**kwargs)[0]
