import os
import tempfile

import nest_asyncio
import streamlit as st
from agentic_rag import get_finance_agent
from agno.media import Image
from agno.agent import Agent
from agno.utils.log import logger
from utils import (
    CUSTOM_CSS,
    about_widget,
    add_message,
    sanitize_agent_text,
    display_tool_calls,
    export_chat_history,
    rename_session_widget,
    session_selector_widget,
    generate_tts_audio,
)

nest_asyncio.apply()
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def restart_agent():
    """Reset the agent and clear chat history"""
    logger.debug("---*--- Restarting agent ---*---")
    st.session_state["agentic_rag_agent"] = None
    st.session_state["agentic_rag_agent_session_id"] = None
    st.session_state["messages"] = []
    # Reset session selector and block one auto-load cycle
    try:
        st.session_state.pop("session_selector", None)
    except Exception:
        pass
    st.session_state["prevent_session_autoload"] = True
    st.rerun()


def main():
    ####################################################################
    # App header
    ####################################################################
    st.markdown("<h1 class='main-title'>Agentic RAG </h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Your intelligent research assistant powered by Agno</p>",
        unsafe_allow_html=True,
    )

    ####################################################################
    # User ID Input (persisted via query params + session state)
    ####################################################################
    initial_uid = ""
    try:
        # Try new API first
        qp = {}
        try:
            qp = dict(st.query_params)  # type: ignore[arg-type]
        except Exception:
            qp = st.experimental_get_query_params()
        if "uid" in qp:
            _val = qp.get("uid")
            initial_uid = _val[0] if isinstance(_val, list) else (_val or "")
    except Exception:
        initial_uid = ""

    # If a uid is present in the URL, always reflect it in session state
    if "cached_user_id" not in st.session_state:
        st.session_state.cached_user_id = initial_uid
    elif initial_uid:
        st.session_state.cached_user_id = initial_uid

    def _persist_uid():
        # Keep URL in sync with current value; remove when empty
        try:
            val = st.session_state.cached_user_id
            if hasattr(st, "query_params"):
                qp = st.query_params
                if val:
                    qp["uid"] = val
                else:
                    try:
                        del qp["uid"]
                    except Exception:
                        pass
            else:
                if val:
                    st.experimental_set_query_params(uid=val)
                else:
                    st.experimental_set_query_params()
        except Exception:
            pass

    user_id = st.sidebar.text_input(
        "Enter your User ID:",
        value=st.session_state.cached_user_id,
        key="cached_user_id",
        help="Enter a unique identifier for your session",
        placeholder="e.g., caribbean_user_123",
        on_change=_persist_uid,
    )

    ####################################################################
    # Initialize Agent
    ####################################################################
    agentic_rag_agent: Agent
    # Only (re)create the agent when the user_id changes AND is non-empty
    if (
        "agentic_rag_agent" not in st.session_state
        or st.session_state["agentic_rag_agent"] is None
        or (
            user_id
            and st.session_state.get("current_user_id") != user_id
        )
    ):
        logger.info("---*--- Creating new Finance Agent  ---*---")
        agentic_rag_agent = get_finance_agent(user_id=user_id or None)
        st.session_state["agentic_rag_agent"] = agentic_rag_agent
        st.session_state["current_user_id"] = user_id
    else:
        agentic_rag_agent = st.session_state["agentic_rag_agent"]

    ####################################################################
    # Load Agent Session from the database
    ####################################################################
    # Check if session ID is already in session state
    session_id_exists = (
        "agentic_rag_agent_session_id" in st.session_state
        and st.session_state["agentic_rag_agent_session_id"]
    )

    if not session_id_exists:
        # Defer session creation until the first user message to avoid empty sessions
        pass
    elif (
        st.session_state["agentic_rag_agent_session_id"]
        and hasattr(agentic_rag_agent, "memory")
        and agentic_rag_agent.memory is not None
        and not agentic_rag_agent.memory.runs
    ):
        # If we have a session ID but no runs, try to load the session explicitly
        try:
            agentic_rag_agent.load_session(
                st.session_state["agentic_rag_agent_session_id"]
            )
        except Exception as e:
            logger.error(f"Failed to load existing session: {str(e)}")
            # Continue anyway

    ####################################################################
    # Load runs from memory
    ####################################################################
    agent_runs = []
    # Only pull runs when we actually have a loaded session id to avoid repopulating after New Chat
    if (
        user_id
        and st.session_state.get("agentic_rag_agent_session_id")
        and not st.session_state.get("prevent_session_autoload")
        and hasattr(agentic_rag_agent, "memory")
        and agentic_rag_agent.memory is not None
    ):
        agent_runs = agentic_rag_agent.memory.runs

    # Initialize messages if it doesn't exist yet
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Only populate messages from agent runs if we haven't already
    if (
        len(st.session_state["messages"]) == 0
        and len(agent_runs) > 0
        and not st.session_state.get("prevent_session_autoload")
    ):
        logger.debug("Loading run history")
        for _run in agent_runs:
            # Check if _run is an object with message attribute
            if hasattr(_run, "message") and _run.message is not None:
                add_message(_run.message.role, _run.message.content)
            # Check if _run is an object with response attribute
            if hasattr(_run, "response") and _run.response is not None:
                add_message("assistant", _run.response.content, _run.response.tools)
    elif len(agent_runs) == 0 and len(st.session_state["messages"]) == 0:
        logger.debug("No run history found")

    if data := st.chat_input("👋 Ask me anything!", accept_file=True):
        # Handle text and file input
        prompt = data.get("text", None)
        uploaded_file = data.get("files", None)
        
        # Ignore empty prompts and also avoid creating sessions if no user_id yet
        if not prompt or not user_id:
            st.warning("Please enter a User ID first.")
            return

        # Add user message with image if uploaded
        add_message("user", prompt, image=uploaded_file)

        # Lazily create/load a chat session upon first user message
        if not st.session_state.get("agentic_rag_agent_session_id"):
            try:
                st.session_state["agentic_rag_agent_session_id"] = (
                    agentic_rag_agent.load_session()
                )
                # Once we create a brand new session, allow autoload again for future reruns
                st.session_state["prevent_session_autoload"] = False
            except Exception as e:
                logger.error(f"Session load error: {str(e)}")
                st.warning("Could not create Agent session, is the database running?")

        # Auto-name the session using the first user message for easier identification
        try:
            first_line = (prompt or "").strip().splitlines()[0][:60]
            if first_line:
                current_name = getattr(agentic_rag_agent, "session_name", None)
                if not current_name or current_name == getattr(agentic_rag_agent, "session_id", current_name):
                    agentic_rag_agent.rename_session(first_line)
        except Exception:
            pass

    # Document Management widgets removed per product direction.
    # Sample Questions removed per product direction.

    ###############################################################
    # Utility buttons
    ###############################################################
    st.sidebar.markdown("#### 🛠️ Utilities")
    col1, col2 = st.sidebar.columns([1, 1])  # Equal width columns
    with col1:
        if st.sidebar.button(
            "🔄 New Chat", use_container_width=True
        ):  # Added use_container_width
            restart_agent()
    with col2:
        if st.sidebar.download_button(
            "💾 Export Chat",
            export_chat_history(),
            file_name="rag_chat_history.md",
            mime="text/markdown",
            use_container_width=True,  # Added use_container_width
        ):
            st.sidebar.success("Chat history exported!")

    ####################################################################
    # Display chat history
    ####################################################################
    for idx, message in enumerate(st.session_state["messages"]):
        if message["role"] in ["user", "assistant"]:
            _content = message["content"]
            if _content is not None:
                with st.chat_message(message["role"]):
                    # Display image if present
                    if message.get("image"):
                        st.image(message["image"], width=200)
                    # Display tool calls if they exist in the message
                    if "tool_calls" in message and message["tool_calls"]:
                        display_tool_calls(st.empty(), message["tool_calls"])
                    st.markdown(_content)
                    # Always show Play for assistant messages; generate audio lazily on first click
                    if message["role"] == "assistant":
                        show_key = f"show_audio_{idx}"
                        if st.button("▶ Play", key=f"play_btn_{idx}"):
                            # Generate audio only if not already present
                            if not message.get("audio_bytes"):
                                with st.spinner("Generating audio..."):
                                    try:
                                        tts_bytes = generate_tts_audio(_content)
                                    except Exception as _tts_err:
                                        tts_bytes = None
                                        st.warning(f"Audio generation failed: {_tts_err}")
                                    else:
                                        st.session_state["messages"][idx]["audio_bytes"] = tts_bytes
                            st.session_state[show_key] = True
                        if st.session_state.get(show_key) and message.get("audio_bytes"):
                            st.audio(message["audio_bytes"], format="audio/mp3")

    ####################################################################
    # Generate response for user message
    ####################################################################
    last_message = (
        st.session_state["messages"][-1] if st.session_state["messages"] else None
    )
    if last_message and last_message.get("role") == "user":
        question = last_message["content"]
        uploaded_image = last_message.get("image")
        
        with st.chat_message("assistant"):
            # Create container for tool calls
            tool_calls_container = st.empty()
            resp_container = st.empty()
            # audio_container used only if user clicks Play; do not auto-render
            audio_container = st.empty()
            response = ""
            try:
                # Handle image file
                images = None
                if uploaded_image:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_image[0].name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_image[0].getvalue())
                        image_path = tmp_file.name
                    images = [Image(filepath=image_path)]
                
                # Run the agent and stream the response
                run_response = agentic_rag_agent.run(question, images=images, stream=True)
                for _resp_chunk in run_response:
                    # Display tool calls if available
                    if hasattr(_resp_chunk, "tool") and _resp_chunk.tool:
                        display_tool_calls(tool_calls_container, [_resp_chunk.tool])

                    # Display response
                    if _resp_chunk.content is not None:
                        response += _resp_chunk.content
                        resp_container.markdown(sanitize_agent_text(response))

                # Final assistant text
                final_text = sanitize_agent_text(response)

                # Save without audio; the Play button will generate on demand
                add_message(
                    "assistant",
                    final_text,
                    agentic_rag_agent.run_response.tools,
                    audio_bytes=None,
                )
                # Rerun so the chat history section picks up the new assistant message with Play button
                st.rerun()
                
                # Clean up temporary image file
                if uploaded_image and 'image_path' in locals():
                    try:
                        os.unlink(image_path)
                    except:
                        pass
                        
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                add_message("assistant", error_message)
                st.error(error_message)
                
                # Clean up temporary image file on error
                if uploaded_image and 'image_path' in locals():
                    try:
                        os.unlink(image_path)
                    except:
                        pass

    ####################################################################
    # Session selector
    ####################################################################
    session_selector_widget(agentic_rag_agent, user_id)
    rename_session_widget(agentic_rag_agent)

    ####################################################################
    # About section
    ####################################################################
    about_widget()


main()
