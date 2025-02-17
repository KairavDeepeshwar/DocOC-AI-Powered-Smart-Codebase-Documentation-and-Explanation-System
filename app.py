import re
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import(
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
st.markdown("""
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.header("Chatbot Settings")
    st.subheader("Model Settings")
    model_name = st.selectbox("Model", ["deepseek-r1:1.5b", "deepseek-r1:3b"],index=0)
    st.divider()
    
st.title("Ollama Chatbot")
st.caption("Ollama is a language model that can generate text based on the input you provide.")



#intiate the engine

llm_engine =ChatOllama(model=model_name,base_url="http://localhost:11434",temperature=0.3)

system_prompt=SystemMessagePromptTemplate.from_template("You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English.")

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role":"ai","content":"Hello, I am Nebby, your AI coding assistant. How can I help you today?"}]

chat_container = st.container()

with chat_container:
    for message in st.session_state.message_log:
       with st.chat_message(message["role"]):
           st.markdown(message["content"])

user_query=st.chat_input("Enter your message here...") 


def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain|llm_engine|StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence= [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"]=="user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        else:
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)


if user_query:
    st.session_state.message_log.append({"role":"user","content":user_query})

    with st.spinner("Thinking..."):
        prompt_chain=build_prompt_chain()
        ai_response=generate_ai_response(prompt_chain=prompt_chain)
    ai_response = re.sub(r"<think>.*?</think>", "", ai_response, flags=re.DOTALL).strip()
    st.session_state.message_log.append({"role":"ai","content":ai_response})

    st.rerun()    