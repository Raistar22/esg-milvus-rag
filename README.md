# 🌿 ESG Report Hybrid RAG

Metadata-filtered RAG over ESG reports using Milvus + BM25 + Flan-T5.

## Metadata Filters
| Filter | Options |
|---|---|
| Sector | Automobile, Finance, Banking, FMCG, ALL |
| Financial Year | 2023-2024, 2024-2025 |
| BRSR Parameter | E1-E7, S1-S7, G1-G7, ALL |

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
