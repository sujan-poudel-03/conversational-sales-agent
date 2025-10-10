from __future__ import annotations

import json
import uuid
from pathlib import Path

import requests
import streamlit as st


st.set_page_config(page_title="Conversational Sales Agent", page_icon="💬", layout="wide")

ORG_BRANCH_OPTIONS: dict[str, list[str]] = {
    "demo-org": ["main-branch", "solar-specialists", "retail-east"],
    "org-a": ["branch-1", "branch-2", "branch-3"],
    "org-b": ["north", "south"],
}
CUSTOM_OPTION = "Custom (manual)"


def _default_context() -> dict[str, str]:
    return {
        "org_id": st.session_state.get("org_id", "demo-org"),
        "branch_id": st.session_state.get("branch_id", "main-branch"),
        "calendar_id": st.session_state.get("calendar_id", ""),
    }


def _call_api(method: str, path: str, payload: dict) -> tuple[int, dict]:
    base_url = st.session_state.get("api_base_url", "http://localhost:8000")
    url = base_url.rstrip("/") + path
    response = requests.request(method, url, json=payload)
    data = {}
    try:
        data = response.json()
    except json.JSONDecodeError:
        pass
    return response.status_code, data


def _ingest_documents() -> None:
    st.header("Knowledge Base Ingestion")
    with st.form("ingest"):
        uploaded_files = st.file_uploader(
            "Upload documents to ingest (optional)",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
        )
        raw_text = st.text_area("Or paste text to ingest", height=150)
        submitted = st.form_submit_button("Run Ingestion")

    if not submitted:
        return

    documents: list[dict] = []
    if uploaded_files:
        uploads_dir = Path.cwd() / ".streamlit_uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        for file in uploaded_files:
            temp_path = uploads_dir / f"upload-{uuid.uuid4().hex}-{file.name}"
            temp_path.write_bytes(file.getbuffer())
            documents.append({"source_path": str(temp_path)})

    if raw_text.strip():
        documents.append({"text": raw_text.strip()})

    if not documents:
        st.warning("Provide at least one document or text snippet.")
        return

    context = _default_context()
    context["user_session_id"] = f"session-{uuid.uuid4().hex}"

    status, data = _call_api(
        "POST",
        "/api/v1/ingest",
        payload={"context": context, "documents": documents},
    )
    if status != 200:
        st.error(f"Ingestion failed ({status}). Response: {data or 'No payload'}")
        return
    st.success(
        f"Ingestion completed: processed={data.get('processed', 0)}, failed={data.get('failed', 0)}"
    )


def _chat_interface() -> None:
    st.header("Chat with the Sales Agent")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for entry in st.session_state["messages"]:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    user_prompt = st.chat_input("Ask a question or request a booking")
    if not user_prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    context = _default_context()
    context["user_session_id"] = f"session-{uuid.uuid4().hex}"
    history = [msg for msg in st.session_state["messages"] if msg["role"] != "system"]

    payload = {
        "context": context,
        "message": {"role": "user", "content": user_prompt},
        "history": history[:-1],
    }

    status, data = _call_api("POST", "/api/v1/chat", payload)
    if status != 200:
        assistant_reply = f"Request failed with status {status}. Response: {data}"
        intent = "error"
    else:
        assistant_reply = data.get("reply", "(no reply)")
        intent = data.get("intent", "?")
        lead_captured = data.get("lead_captured", False)
        appointment_id = data.get("appointment_id")
        metadata_bits = [f"intent: {intent}", f"lead_captured: {lead_captured}"]
        if appointment_id:
            metadata_bits.append(f"appointment_id: {appointment_id}")
        assistant_reply += f"\n\n_{', '.join(metadata_bits)}_"

    st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)


def _sidebar_controls() -> None:
    st.sidebar.title("Session Settings")
    st.session_state["api_base_url"] = st.sidebar.text_input(
        "API base URL",
        st.session_state.get("api_base_url", "http://localhost:8000"),
    )

    org_options = sorted(ORG_BRANCH_OPTIONS.keys())
    current_org = st.session_state.get("org_id", org_options[0])
    org_select_options = org_options + [CUSTOM_OPTION]
    org_index = org_select_options.index(current_org) if current_org in org_options else len(org_select_options) - 1
    selected_org = st.sidebar.selectbox("Organization", org_select_options, index=org_index)

    if selected_org == CUSTOM_OPTION:
        custom_org_default = current_org if current_org not in org_options else ""
        org_value = st.sidebar.text_input("Org ID", custom_org_default or "demo-org")
    else:
        org_value = selected_org
    st.session_state["org_id"] = org_value

    branch_candidates = ORG_BRANCH_OPTIONS.get(org_value, ORG_BRANCH_OPTIONS.get(selected_org, []))
    branch_candidates = sorted(set(branch_candidates))
    branch_select_options = (branch_candidates + [CUSTOM_OPTION]) if branch_candidates else [CUSTOM_OPTION]
    current_branch = st.session_state.get("branch_id", branch_candidates[0] if branch_candidates else "")
    if current_branch and current_branch in branch_candidates:
        branch_index = branch_select_options.index(current_branch)
    else:
        branch_index = len(branch_select_options) - 1

    selected_branch = st.sidebar.selectbox("Branch", branch_select_options, index=branch_index)
    if selected_branch == CUSTOM_OPTION:
        default_branch = current_branch if current_branch not in branch_candidates else ""
        branch_value = st.sidebar.text_input("Branch ID", default_branch or "main-branch")
    else:
        branch_value = selected_branch
    st.session_state["branch_id"] = branch_value

    st.session_state["calendar_id"] = st.sidebar.text_input(
        "Calendar ID (optional)",
        st.session_state.get("calendar_id", ""),
    )
    if st.sidebar.button("Reset conversation"):
        st.session_state["messages"] = []
        st.experimental_rerun()


def main() -> None:
    _sidebar_controls()
    tab_ingest, tab_chat = st.tabs(["Ingest KB", "Chat"])
    with tab_ingest:
        _ingest_documents()
    with tab_chat:
        _chat_interface()


if __name__ == "__main__":
    main()
