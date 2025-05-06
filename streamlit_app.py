import uuid

import lancedb
import streamlit as st
from dotenv import load_dotenv
from lancedb.rerankers import RRFReranker
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import OpenAI

# Generate a UUIDv4
if "session_uuid" not in st.session_state:
    st.session_state.session_uuid = str(uuid.uuid4())
session_uuid = st.session_state.session_uuid

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

reranker = RRFReranker()

langfuse = Langfuse()


@st.cache_resource
def init_db():
    """Initialize database connection.

    Returns:
        LanceDB table object
    """
    db = lancedb.connect("db/lancedb")
    return db.open_table("molrag")


@observe()
def get_context(query: str, table, num_results: int = 8) -> str:
    """Search the database for relevant context.

    Args:
        query: User's question
        table: LanceDB table object
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    # results = table.search(query).limit(num_results).to_pandas()
    results = (
        table.search(
            query,
            query_type="hybrid",
            vector_column_name="vector",
            fts_columns="text",
        )
        .rerank(reranker)
        .limit(num_results)
        .to_pandas()
    )

    contexts = []

    for _, row in results.iterrows():
        context = "[DOCUMENT]"

        url = row["metadata"]["url"]
        title = row["metadata"]["page_title"]
        source = row["metadata"]["source"]

        if title:
            context += f"\nTitle: {title}"

        if source:
            context += f"\nSource: {source}"

        if url:
            context += f"\nURL: {url}"

        contexts.append(context + "\nContent: " + row["text"] + "\n[/DOCUMENT]\n")

    return ["\n\n".join(contexts), results]


@observe
def get_chat_response(messages, context: str) -> str:
    """Get streaming response from OpenAI API.

    Args:
        messages: Chat history
        context: Retrieved context from database

    Returns:
        str: Model's response
    """
    langfuse_prompt = langfuse.get_prompt("Simple Q&A prompt")
    compiled_prompt = langfuse_prompt.compile(context=context)

    messages_with_context = [{"role": "system", "content": compiled_prompt}, *messages]

    # Create the streaming response
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages_with_context,
        temperature=0.7,
        stream=True,
        langfuse_prompt=langfuse_prompt,  # capture used prompt version in trace
    )

    # Use Streamlit's built-in streaming capability
    response = st.write_stream(stream)
    return response


@observe()
def handle_prompt(prompt: str, table):
    """Handles the user prompt, gets context, generates response, and updates chat."""

    langfuse_context.update_current_trace(session_id=session_uuid, input=prompt)

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get relevant context
    with st.status("Searching documentation...", expanded=False):
        context_data = get_context(prompt, table)
        context = context_data[0]
        results = context_data[1]
        st.markdown(
            """
            <style>
            .search-result {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                background-color: #f0f2f6;
            }
            .search-result summary {
                cursor: pointer;
                color: #0f52ba;
                font-weight: 500;
            }
            .search-result summary:hover {
                color: #1e90ff;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                font-style: italic;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        st.write("Found relevant sections:")

        for _, row in results.iterrows():
            st.markdown(
                f"""
              <div class="search-result">
                  <details>
                      <summary>{row["metadata"]["page_title"]}</summary>
                      <div class="metadata">Source: {row["metadata"]["source"]}</div>
                      <div class="metadata">URL: <a target='_blank 'href='{row["metadata"]["url"]}'>Click to open in new tab</a></div>
                      <div class="metadata">Header 1: {row["metadata"]["header_1"]}</div>
                      <div class="metadata">Header 2: {row["metadata"]["header_2"]}</div>
                      <div class="metadata">Header 3: {row["metadata"]["header_3"]}</div>
                      <div class="metadata">Header 3: {row["metadata"]["header_4"]}</div>
                      <div style="margin-top: 8px;">{row["text"]}</div>
                  </details>
              </div>
          """,
                unsafe_allow_html=True,
            )

    # Display assistant response first
    with st.chat_message("assistant"):
        # Get model response with streaming
        response = get_chat_response(st.session_state.messages, context)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    return response


# Initialize Streamlit app
st.title("📚 Molecule Q&A")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize database connection
table = init_db()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about DeSci or Molecule"):
    handle_prompt(prompt, table)
