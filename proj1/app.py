# app.py
import os
from dotenv import load_dotenv
import streamlit as st

# LangChain + LangGraph
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, DuckDuckGoSearchAPIWrapper
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated
from langgraph.graph.message import add_messages

# -----------------------
# ENV + API Keys
# -----------------------
load_dotenv()
os.environ["GroqApi"] = os.getenv("GROQ_API_KEY", "")
os.environ["TavilyApi"] = os.getenv("TAVILY_API_KEY", "")

# -----------------------
# Tools setup
# -----------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=1000)
wikipedia_wrapper = WikipediaAPIWrapper(
    top_k_results=3, doc_content_chars_max=1000)
duckduckgo_wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)

arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
duckduckgo_tool = DuckDuckGoSearchRun(api_wrapper=duckduckgo_wrapper)

tools = [arxiv_tool, wikipedia_tool]

# -----------------------
# LangGraph State
# -----------------------


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    mode: str


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="LangGraph Chatbot",
                   page_icon="🤖", layout="wide")
st.title("🤖 LangGraph Chatbot with Groq + Tools")

# Sidebar
st.sidebar.header("⚙️ Settings")
api_key = st.sidebar.text_input(
    "🔑 Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
model = st.sidebar.selectbox(
    "📌 Select Model",
    ["llama-3.1-8b-instant", "openai/gpt-oss-20b",
        "deepseek-r1-distill-llama-70b", "qwen/qwen3-32b"],
    index=0
)
mode = st.sidebar.radio("💡 Usecase", ["normal", "news", "tools"], index=0)

# LLM init
llm = ChatGroq(model=model, groq_api_key=api_key if api_key else None)
llm_with_tools = llm.bind_tools(tools)

# -----------------------
# Node definitions
# -----------------------


def tool_calling_llm(state: State):
    res = llm_with_tools.invoke(state["messages"])
    return {"messages": [res]}


def news_llm(state: State):
    ip = state["messages"][0].content
    res = duckduckgo_tool.invoke(ip)
    return {"messages": [res]}


def normal_llm(state: State):
    res = llm.invoke(state["messages"])
    return {"messages": [res]}

# Router


def usecase_condition(state):
    if state.get("mode") == "tools":
        return "tool_calling_llm"
    elif state.get("mode") == "news":
        return "news"
    else:
        return "tool_calling_llm_normal"


# -----------------------
# Build graph
# -----------------------
builder = StateGraph(State)

builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode(tools))
builder.add_node("news", news_llm)
builder.add_node("tool_calling_llm_normal", normal_llm)

builder.add_conditional_edges(
    START,
    usecase_condition,
    {
        "tool_calling_llm": "tool_calling_llm",
        "news": "news",
        "tool_calling_llm_normal": "tool_calling_llm_normal",
    }
)

builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", "tool_calling_llm")

builder.add_edge("news", END)
builder.add_edge("tool_calling_llm_normal", END)

graph = builder.compile()

# -----------------------
# Chat UI
# -----------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    # Add user message
    st.session_state["messages"].append(
        {"role": "user", "content": user_input})

    # Run graph
    state = {"messages": st.session_state["messages"], "mode": mode}
    result = graph.invoke(state)

    assistant_reply = result["messages"][-1].content
    combined_reply = f"**Q:** {user_input}\n\n**A:** {assistant_reply}"

    st.session_state["messages"].append(
        {"role": "assistant", "content": combined_reply})
    st.chat_message("assistant").markdown(combined_reply)
