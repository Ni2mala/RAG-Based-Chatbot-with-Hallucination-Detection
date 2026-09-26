import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

st.set_page_config(page_title="RAG-Guard Banking Chatbot")

API_URL = st.sidebar.text_input("API base URL", value="http://127.0.0.1:8000")


def ask(question: str) -> dict:
    resp = requests.post(
        f"{API_URL}/chat",
        json={"question": question},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


st.title("RAG-Guard Banking Chatbot")
st.caption("Answers only from approved banking documents, with hallucination checks.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("refused"):
            st.warning(msg["content"])
        for s in msg.get("sources", []):
            st.markdown(
                f"- {s['intent']} (category {s['category']}) - similarity {s['score']:.2f}"
            )
if msg.get("status"):
            parts = [f"**Verification:** `{msg['status']}`"]
            if msg.get("nli_grounding") is not None:
                parts.append(f"NLI: {msg['nli_grounding']:.2f}")
            if msg.get("llm_grounding") is not None:
                parts.append(f"LLM-judge: {msg['llm_grounding']:.2f}")
            st.markdown(" &nbsp;·&nbsp; ".join(parts))
            if msg.get("unsupported"):
                n = len(msg["unsupported"])
                st.markdown(f"Unsupported claims detected: {n}")

if prompt := st.chat_input("Ask a banking question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state["messages"].append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data = ask(prompt)
        answer = data["answer"]
        if data.get("refused"):
            st.warning(answer)
        else:
            st.markdown(answer)

        sources = data.get("sources", [])
        for s in sources:
            st.markdown(
                f"- {s['intent']} (category {s['category']}) - similarity {s['score']:.2f}"
            )

v = data.get("verification", {})
        status = v.get("status", "n/a")
        rel = v.get("retrieval_relevance", 0.0)
        nli = v.get("nli_grounding")
        llm = v.get("llm_grounding")
        line = f"**Verification:** `{status}` (relevance {rel:.2f})"
        if nli is not None:
            line += f" · NLI: {nli:.2f}"
        if llm is not None:
            line += f" · LLM-judge: {llm:.2f}"
        st.markdown(line)
        unsupported = v.get("unsupported_claims") or []
        if unsupported:
            st.markdown(f"Unsupported claims detected: {len(unsupported)}")

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "status": status,
            "nli_grounding": nli,
            "llm_grounding": llm,
            "unsupported": unsupported,
        }
    )
