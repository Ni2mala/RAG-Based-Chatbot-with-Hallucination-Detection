import os

import requests
import streamlit as st

st.set_page_config(page_title="RAG-Guard Banking Chatbot")

# Fixed from the environment, not user-editable, so the server can't be pointed at arbitrary URLs.
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")


def ask(question: str) -> dict:
    resp = requests.post(f"{API_URL}/chat", json={"question": question}, timeout=180)
    resp.raise_for_status()
    return resp.json()


def render_assistant(msg: dict) -> None:
    if msg.get("refused"):
        st.warning(msg["content"])
    else:
        st.markdown(msg["content"])

    for s in msg.get("sources", []):
        st.markdown(
            f"- {s['intent']} (category {s['category']}) - similarity {s['score']:.2f}"
        )

    line = f"**Verification:** `{msg.get('status', 'n/a')}` (relevance {msg.get('relevance', 0.0):.2f})"
    if msg.get("nli_grounding") is not None:
        line += f" · NLI: {msg['nli_grounding']:.2f}"
    if msg.get("llm_grounding") is not None:
        line += f" · LLM-judge: {msg['llm_grounding']:.2f}"
    st.markdown(line)
    if msg.get("unsupported"):
        st.markdown(f"Unsupported claims detected: {len(msg['unsupported'])}")


st.title("RAG-Guard Banking Chatbot")
st.caption("Answers only from approved banking documents, with hallucination checks.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant(msg)
        else:
            st.markdown(msg["content"])

if prompt := st.chat_input("Ask a banking question..."):
    st.chat_message("user").markdown(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                data = ask(prompt)
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
        else:
            v = data.get("verification", {})
            reply = {
                "role": "assistant",
                "content": data["answer"],
                "refused": data.get("refused", False),
                "sources": data.get("sources", []),
                "status": v.get("status", "n/a"),
                "relevance": v.get("retrieval_relevance", 0.0),
                "nli_grounding": v.get("nli_grounding"),
                "llm_grounding": v.get("llm_grounding"),
                "unsupported": v.get("unsupported_claims") or [],
            }
            render_assistant(reply)
            st.session_state["messages"].append(reply)
