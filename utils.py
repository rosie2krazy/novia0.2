from typing import Any, Dict, List, Optional
import re
import os

import streamlit as st
from agentic_rag import get_finance_agent
from agno.agent import Agent
from agno.models.response import ToolExecution
from agno.utils.log import logger
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs


# Regex to capture snippets like:
#   finance_agent: some_tool(arg=1) completed in 0.23s
# The pattern below matches optional "agent: " prefix, a tool/function call, and the trailing timing text
TOOL_SNIPPET = re.compile(
    r"(?:[A-Za-z_][\w-]*\s*:\s*)?[A-Za-z_][\w-]*\([^)]*\)\s*completed\s+in\s+[0-9.]+s\.?",
    re.IGNORECASE,
)


def sanitize_agent_text(text: str) -> str:
    """Remove tool-call debug fragments like `get_user_transactions(... ) completed in Xs`.

    Keeps the rest of the message intact.
    """
    if not text:
        return text
    # Remove exact tool-call snippets
    text = TOOL_SNIPPET.sub("", text)
    # Remove a colon followed by tool-call snippet
    text = re.sub(
        r":\s*[a-zA-Z_]\w*\([^)]*\)\s+completed\s+in\s+[0-9.]+s\.?",
        ": ",
        text,
        flags=re.IGNORECASE,
    )
    # Remove dangling `completed in Xs` if any remain
    text = re.sub(r"\s*completed\s+in\s+[0-9.]+s\.?", "", text, flags=re.IGNORECASE)
    return text.strip()


def add_message(
    role: str,
    content: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    image=None,
    audio_bytes: Optional[bytes] = None,
) -> None:
    """Safely add a message to the session state"""
    if "messages" not in st.session_state or not isinstance(
        st.session_state["messages"], list
    ):
        st.session_state["messages"] = []
    
    message = {"role": role, "content": content, "tool_calls": tool_calls}
    if image:
        message["image"] = image
    if audio_bytes is not None:
        message["audio_bytes"] = audio_bytes
        
    st.session_state["messages"].append(message)


def export_chat_history():
    """Export chat history as markdown"""
    if "messages" in st.session_state:
        chat_text = "# Finance Agent - Chat History\n\n"
        for msg in st.session_state["messages"]:
            role = "🤖 Assistant" if msg["role"] == "agent" else "👤 User"
            chat_text += f"### {role}\n{msg['content']}\n\n"
            if msg.get("tool_calls"):
                chat_text += "#### Tools Used:\n"
                for tool in msg["tool_calls"]:
                    if isinstance(tool, dict):
                        tool_name = tool.get("name", "Unknown Tool")
                    else:
                        tool_name = getattr(tool, "name", "Unknown Tool")
                    chat_text += f"- {tool_name}\n"
        return chat_text
    return ""


def display_tool_calls(tool_calls_container, tools: List[ToolExecution]):
    """Display tool calls in a streamlit container with expandable sections.

    Args:
        tool_calls_container: Streamlit container to display the tool calls
        tools: List of tool call dictionaries containing name, args, content, and metrics
    """
    if not tools:
        return

    with tool_calls_container.container():
        for tool_call in tools:
            # Normalize ToolExecution or dict-like structure
            if hasattr(tool_call, "tool_name"):
                _tool_name = getattr(tool_call, "tool_name", None) or "Unknown Tool"
                _tool_args = getattr(tool_call, "tool_args", None) or {}
                _content = getattr(tool_call, "result", None) or ""
                _metrics = getattr(tool_call, "metrics", None) or {}
            elif isinstance(tool_call, dict):
                _tool_name = (
                    tool_call.get("tool_name")
                    or tool_call.get("name")
                    or tool_call.get("tool")
                    or "Unknown Tool"
                )
                _tool_args = tool_call.get("tool_args") or tool_call.get("args") or {}
                _content = tool_call.get("result") or tool_call.get("content") or ""
                _metrics = tool_call.get("metrics") or {}
            else:
                _tool_name = str(tool_call)
                _tool_args = {}
                _content = ""
                _metrics = {}

            # Safely create the title with a default if tool name is None
            title = f"🛠️ {_tool_name.replace('_', ' ').title() if _tool_name else 'Tool Call'}"

            with st.expander(title, expanded=False):
                if isinstance(_tool_args, dict) and "query" in _tool_args:
                    st.code(_tool_args["query"], language="sql")
                # Handle string arguments
                elif isinstance(_tool_args, str) and _tool_args:
                    try:
                        # Try to parse as JSON
                        import json

                        args_dict = json.loads(_tool_args)
                        st.markdown("**Arguments:**")
                        st.json(args_dict)
                    except:
                        # If not valid JSON, display as string
                        st.markdown("**Arguments:**")
                        st.markdown(f"```\n{_tool_args}\n```")
                # Handle dict arguments
                elif _tool_args and _tool_args != {"query": None}:
                    st.markdown("**Arguments:**")
                    st.json(_tool_args)

                if _content:
                    st.markdown("**Results:**")
                    if isinstance(_content, (dict, list)):
                        st.json(_content)
                    else:
                        try:
                            st.json(_content)
                        except Exception:
                            st.markdown(_content)

                if _metrics:
                    st.markdown("**Metrics:**")
                    st.json(
                        _metrics if isinstance(_metrics, dict) else _metrics.to_dict()
                    )


def rename_session_widget(agent: Agent) -> None:
    """Rename the current session of the agent and save to storage"""

    container = st.sidebar.container()

    # Initialize session_edit_mode if needed
    if "session_edit_mode" not in st.session_state:
        st.session_state.session_edit_mode = False

    if st.sidebar.button("✎ Rename Session"):
        st.session_state.session_edit_mode = True
        st.rerun()

    if st.session_state.session_edit_mode:
        new_session_name = st.sidebar.text_input(
            "Enter new name:",
            value=agent.session_name,
            key="session_name_input",
        )
        if st.sidebar.button("Save", type="primary"):
            if new_session_name:
                agent.rename_session(new_session_name)
                st.session_state.session_edit_mode = False
                st.rerun()


def session_selector_widget(agent: Agent, user_id: str) -> None:
    """Display a session selector in the sidebar"""

    if agent.storage:
        # Filter sessions by user_id
        agent_sessions = agent.storage.get_all_sessions()
        user_sessions = [session for session in agent_sessions if session.user_id == user_id]
        # print(f"User {user_id} sessions: {user_sessions}")

        session_options = []
        for session in user_sessions:
            session_id = session.session_id
            session_name = (
                session.session_data.get("session_name", None)
                if session.session_data
                else None
            )
            display_name = session_name if session_name else session_id
            session_options.append({"id": session_id, "display": display_name})

        if session_options:
            placeholder_display = "— New Chat —"
            display_list = [placeholder_display] + [s["display"] for s in session_options]

            # Decide default index
            default_index = 0
            current_session_id = st.session_state.get("agentic_rag_agent_session_id")
            if not st.session_state.get("prevent_session_autoload") and current_session_id:
                try:
                    current_index = next(
                        i for i, s in enumerate(session_options) if s["id"] == current_session_id
                    )
                    default_index = current_index + 1  # offset due to placeholder at 0
                except StopIteration:
                    default_index = 0

            selected_display = st.sidebar.selectbox(
                "Session",
                options=display_list,
                index=default_index,
                key="session_selector",
            )

            # Map display back to id (None if placeholder)
            if selected_display == placeholder_display:
                selected_session_id = None
            else:
                selected_session_id = next(
                    s["id"] for s in session_options if s["display"] == selected_display
                )

            # If placeholder is selected, do nothing
            if selected_session_id is None:
                return

            if (
                st.session_state.get("agentic_rag_agent_session_id")
                != selected_session_id
            ):
                logger.info(
                    f"---*--- Loading {user_id} run: {selected_session_id} ---*---"
                )

                try:
                    new_agent = get_finance_agent(
                        user_id=user_id,
                        session_id=selected_session_id,
                    )

                    st.session_state["agentic_rag_agent"] = new_agent
                    st.session_state["agentic_rag_agent_session_id"] = (
                        selected_session_id
                    )
                    st.session_state["prevent_session_autoload"] = False

                    st.session_state["messages"] = []

                    selected_session_obj = next(
                        (
                            s
                            for s in user_sessions
                            if s.session_id == selected_session_id
                        ),
                        None,
                    )

                    if (
                        selected_session_obj
                        and selected_session_obj.memory
                        and "runs" in selected_session_obj.memory
                    ):
                        seen_messages = set()

                        for run in selected_session_obj.memory["runs"]:
                            if "messages" in run:
                                for msg in run["messages"]:
                                    msg_role = msg.get("role")
                                    msg_content = msg.get("content")

                                    if not msg_content or msg_role == "system":
                                        continue

                                    msg_id = f"{msg_role}:{msg_content}"

                                    if msg_id in seen_messages:
                                        continue

                                    seen_messages.add(msg_id)

                                    if msg_role == "assistant":
                                        tool_calls = None
                                        if "tool_calls" in msg:
                                            tool_calls = msg["tool_calls"]
                                        elif "metrics" in msg and msg.get("metrics"):
                                            tools = run.get("tools")
                                            if tools:
                                                tool_calls = tools

                                        add_message(msg_role, msg_content, tool_calls)
                                    else:
                                        add_message(msg_role, msg_content)

                            elif (
                                "message" in run
                                and isinstance(run["message"], dict)
                                and "content" in run["message"]
                            ):
                                user_msg = run["message"]["content"]
                                msg_id = f"user:{user_msg}"

                                if msg_id not in seen_messages:
                                    seen_messages.add(msg_id)
                                    add_message("user", user_msg)

                                if "content" in run and run["content"]:
                                    asst_msg = run["content"]
                                    msg_id = f"assistant:{asst_msg}"

                                    if msg_id not in seen_messages:
                                        seen_messages.add(msg_id)
                                        add_message(
                                            "assistant", asst_msg, run.get("tools")
                                        )

                    st.rerun()
                except Exception as e:
                    logger.error(f"Error switching sessions: {str(e)}")
                    st.sidebar.error(f"Error loading session: {str(e)}")
        else:
            st.sidebar.info("No saved sessions available.")


def about_widget() -> None:
    """Display an about section in the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This Finance Agent helps you analyze stocks and get market insights using AI-powered analysis.

    Built with:
    - 🚀 Agno
    - 💫 Streamlit
    - 📊 YFinance
    """)


CUSTOM_CSS = """
    <style>
    /* Main Styles */
   .main-title {
        text-align: center;
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: bold;
        padding: 1em 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2em;
    }
    .stButton button {
        width: 100%;
        border-radius: 20px;
        margin: 0.2em 0;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .chat-container {
        border-radius: 15px;
        padding: 1em;
        margin: 1em 0;
        background-color: #f5f5f5;
    }
    .tool-result {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1em;
        margin: 1em 0;
        border-left: 4px solid #3B82F6;
    }
    .status-message {
        padding: 1em;
        border-radius: 10px;
        margin: 1em 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
    }
    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        .chat-container {
            background-color: #2b2b2b;
        }
        .tool-result {
            background-color: #1e1e1e;
        }
    }
    </style>
"""


# -----------------------------
# ElevenLabs TTS Integration
# -----------------------------
_ELEVENLABS_CLIENT: Optional[ElevenLabs] = None


def _get_elevenlabs_client() -> ElevenLabs:
    """Return a cached ElevenLabs client using the ELEVENLABS_API_KEY from the environment."""
    global _ELEVENLABS_CLIENT
    if _ELEVENLABS_CLIENT is None:
        load_dotenv()
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set in the environment")
        _ELEVENLABS_CLIENT = ElevenLabs(api_key=api_key)
    return _ELEVENLABS_CLIENT


def generate_tts_audio(
    text: str,
    *,
    voice_id: str = "mrDMz4sYNCz18XYFpmyV",  # user-provided working voice id
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
) -> bytes:
    """Generate MP3 audio bytes for the given text using ElevenLabs.

    Returns raw MP3 bytes suitable for `st.audio`.
    """
    if not text:
        return b""
    client = _get_elevenlabs_client()
    response = client.text_to_speech.convert(
        voice_id=voice_id,
        output_format=output_format,
        text=text,
        model_id=model_id,
    )
    chunks: List[bytes] = []
    for chunk in response:
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)
