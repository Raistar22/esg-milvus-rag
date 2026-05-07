import streamlit as st
import tempfile, os
from backend.milvus_store  import ingest_pdf
from backend.bm25_index    import build_bm25
from backend.hybrid_search import hybrid_search
from backend.llm           import run_llm
from backend.metadata      import SECTORS, FINANCIAL_YEARS, BRSR_PARAMETERS

st.set_page_config(page_title="ESG Report RAG", page_icon="🌿", layout="wide")
st.title("🌿 ESG Report — Hybrid Search RAG")
st.caption("Milvus + BM25 + Flan-T5 | Metadata Filtering by Sector / FY / BRSR")

if "ingested"     not in st.session_state: st.session_state.ingested     = False
if "chat_history" not in st.session_state: st.session_state.chat_history = []

with st.sidebar:
    st.header("📄 Document")
    uploaded = st.file_uploader("Upload ESG PDF", type=["pdf"])

    if uploaded and not st.session_state.ingested:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        bar = st.progress(0, text="Starting...")
        with st.spinner("Ingesting..."):
            n, docs = ingest_pdf(tmp_path, progress_cb=lambda p,m: bar.progress(p, text=m))
            build_bm25(docs)
        os.unlink(tmp_path)
        st.session_state.ingested = True
        bar.progress(1.0, "✅ Done!")
        st.success(f"Ingested {n} chunks")

    st.divider()
    st.header("🔍 Metadata Filters")
    sector         = st.selectbox("Sector",         SECTORS,                          index=SECTORS.index("ALL"))
    financial_year = st.selectbox("Financial Year", FINANCIAL_YEARS,                  index=0)
    brsr_parameter = st.selectbox("BRSR Parameter", ["ALL"]+BRSR_PARAMETERS[:-1],     index=0)
    st.divider()
    st.header("⚖️ Hybrid Weights")
    dense_weight = st.slider("Dense (Semantic)", 0.0, 1.0, 0.6, 0.1)
    bm25_weight  = st.slider("BM25 (Keyword)",   0.0, 1.0, 0.4, 0.1)
    top_k        = st.slider("Top-K chunks",     1,   10,  5)
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

if not st.session_state.ingested:
    st.info("👈 Upload an ESG PDF from the sidebar to get started.")
    st.stop()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "chunks" in msg:
            with st.expander(f"📚 Sources ({len(msg['chunks'])})"):
                for i, c in enumerate(msg["chunks"], 1):
                    st.markdown(f"**[{i}]** Score:`{c['fused_score']:.4f}` | Page`{c['page_number']}` | via`{c['source']}` | `{c['sector']}` | `{c['financial_year']}` | `{c['brsr_parameter']}`")
                    st.caption(c["text"][:300]+"...")
                    st.divider()

question = st.chat_input("Ask about the ESG report...")
if question:
    st.session_state.chat_history.append({"role":"user","content":question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching + generating..."):
            chunks, filter_expr = hybrid_search(
                question, sector, financial_year, brsr_parameter,
                top_k, dense_weight, bm25_weight
            )
            if not chunks:
                answer = "⚠️ No relevant chunks found. Try relaxing filters."
            else:
                context = "\n\n".join([
                    f"[Chunk {i}|Page {c['page_number']}|{c['sector']}|{c['financial_year']}|BRSR:{c['brsr_parameter']}]\n{c['text']}"
                    for i,c in enumerate(chunks,1)
                ])
                answer = run_llm(f"""You are an ESG analyst. Answer using ONLY the context below.
If not found say "Not found in the report."

Context:
{context[:3500]}

Question: {question}
Answer:""")
        st.markdown(answer)
        if filter_expr: st.caption(f"🧮 Filter: `{filter_expr}`")
        with st.expander(f"📚 Sources ({len(chunks)})"):
            for i,c in enumerate(chunks,1):
                st.markdown(f"**[{i}]** Score:`{c['fused_score']:.4f}` | Page`{c['page_number']}` | via`{c['source']}` | `{c['sector']}` | `{c['financial_year']}` | `{c['brsr_parameter']}`")
                st.caption(c["text"][:300]+"...")
                st.divider()
    st.session_state.chat_history.append({"role":"assistant","content":answer,"chunks":chunks})
