import streamlit as st
import requests
import uuid
import os
from typing import List, Dict, Any

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AutoDrive - Assistant RAG Multi-Agents",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .session-info {
        background-color: #f3f4f6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .source-tag {
        display: inline-block;
        background-color: #e0e7ff;
        color: #3730a3;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        margin: 0.25rem;
    }
    .feedback-btn {
        margin: 0.25rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "pending_feedback" not in st.session_state:
        st.session_state.pending_feedback = {}


def call_api_chat(question: str, session_id: str = None) -> Dict[str, Any]:
    try:
        payload = {"question": question}
        if session_id:
            payload["session_id"] = session_id
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de communication avec l'API: {e}")
        return None


def call_api_history(session_id: str) -> List[Dict[str, str]]:
    try:
        response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/historique", timeout=10)
        response.raise_for_status()
        return response.json().get("messages", [])
    except requests.exceptions.RequestException:
        return []


def call_api_feedback(session_id: str, question: str, reponse: str, utile: bool) -> bool:
    try:
        payload = {
            "session_id": session_id,
            "question": question,
            "reponse": reponse,
            "utile": utile
        }
        response = requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=10)
        response.raise_for_status()
        return response.json().get("success", False)
    except requests.exceptions.RequestException:
        return False


def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def new_conversation():
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.pending_feedback = {}
    st.rerun()


def load_history():
    if st.session_state.session_id:
        history = call_api_history(st.session_state.session_id)
        st.session_state.messages = []
        for msg in history:
            st.session_state.messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })


def main():
    init_session_state()

    with st.sidebar:
        st.markdown('<div class="main-header">🚗 AutoDrive</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Assistant RAG Multi-Agents</div>', unsafe_allow_html=True)

        st.divider()

        st.markdown("### Session en cours")
        if st.session_state.session_id:
            short_id = st.session_state.session_id[:8] + "..."
            st.markdown(f'<div class="session-info"><strong>ID:</strong> {short_id}<br><strong>Messages:</strong> {len([m for m in st.session_state.messages if m["role"] == "user"])}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="session-info"><em>Nouvelle conversation</em></div>', unsafe_allow_html=True)

        st.button("🆕 Nouvelle conversation", on_click=new_conversation, use_container_width=True)

        st.divider()

        st.markdown("### Options")
        st.session_state.show_sources = st.checkbox("Afficher les sources", value=st.session_state.show_sources)

        st.divider()

        st.markdown("### Statut API")
        if check_api_health():
            st.success("✅ API connectée (port 8000)")
        else:
            st.error("❌ API inaccessible")

    st.markdown('<div class="main-header">💬 Chat AutoDrive</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Posez vos questions sur les contrats, assurances, véhicules, maintenance ou FAQ</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.info("👋 Bienvenue ! Commencez par poser une question ou cliquez sur 'Nouvelle conversation' dans la barre latérale.")

    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant" and "metadata" in message:
                meta = message["metadata"]
                if st.session_state.show_sources and meta.get("sources"):
                    st.markdown("**📚 Sources :**")
                    for src in meta["sources"]:
                        st.markdown(f'<span class="source-tag">{src}</span>', unsafe_allow_html=True)

                st.markdown(f"*Agent utilisé : {meta.get('agent_utilise', 'N/A')}*")

                msg_key = f"feedback_{idx}"
                if msg_key not in st.session_state.pending_feedback:
                    col1, col2, _ = st.columns([1, 1, 10])
                    with col1:
                        if st.button("👍", key=f"up_{idx}", help="Utile"):
                            success = call_api_feedback(
                                st.session_state.session_id or "",
                                st.session_state.messages[idx - 1]["content"] if idx > 0 else "",
                                message["content"],
                                True
                            )
                            if success:
                                st.session_state.pending_feedback[msg_key] = "positive"
                                st.toast("Merci pour votre feedback ! 👍")
                                st.rerun()
                    with col2:
                        if st.button("👎", key=f"down_{idx}", help="Pas utile"):
                            success = call_api_feedback(
                                st.session_state.session_id or "",
                                st.session_state.messages[idx - 1]["content"] if idx > 0 else "",
                                message["content"],
                                False
                            )
                            if success:
                                st.session_state.pending_feedback[msg_key] = "negative"
                                st.toast("Merci pour votre feedback ! 👎")
                                st.rerun()
                else:
                    feedback_val = st.session_state.pending_feedback[msg_key]
                    if feedback_val == "positive":
                        st.success("👍 Merci !")
                    else:
                        st.error("👎 Noté")

    prompt = st.chat_input("Votre question sur AutoDrive...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Réflexion en cours..."):
                result = call_api_chat(prompt, st.session_state.session_id)

            if result:
                st.session_state.session_id = result["session_id"]
                st.write(result["reponse"])

                if st.session_state.show_sources and result["sources"]:
                    st.markdown("**📚 Sources :**")
                    for src in result["sources"]:
                        st.markdown(f'<span class="source-tag">{src}</span>', unsafe_allow_html=True)

                st.markdown(f"*Agent utilisé : {result['agent_utilise']}*")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["reponse"],
                    "metadata": {
                        "sources": result["sources"],
                        "agent_utilise": result["agent_utilise"]
                    }
                })

                col1, col2, _ = st.columns([1, 1, 10])
                with col1:
                    if st.button("👍", key=f"up_new_{len(st.session_state.messages)}", help="Utile"):
                        success = call_api_feedback(
                            st.session_state.session_id,
                            prompt,
                            result["reponse"],
                            True
                        )
                        if success:
                            st.toast("Merci pour votre feedback ! 👍")
                with col2:
                    if st.button("👎", key=f"down_new_{len(st.session_state.messages)}", help="Pas utile"):
                        success = call_api_feedback(
                            st.session_state.session_id,
                            prompt,
                            result["reponse"],
                            False
                        )
                        if success:
                            st.toast("Merci pour votre feedback ! 👎")
            else:
                st.error("Erreur lors de la génération de la réponse. Vérifiez que l'API est démarrée.")
                st.session_state.messages.pop()


if __name__ == "__main__":
    main()