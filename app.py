import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import(
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from visual import streamlit_ui
from codeusage import  visualisation


#intialise the env

from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
import os


# Streamlit Page Config
st.set_page_config(
    page_title="Smart Documentation & Code Explanation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS-only background with Matisse blue styling
def add_bg_gradient():
    st.markdown(
        """
        <style>
        /* Create a dark gradient background */
        .stApp {
            background: linear-gradient(to bottom right, #1e1e2f, #000000);
        }
        
        /* Add a subtle animated pattern overlay */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(circle at 25% 25%, rgba(75, 140, 222, 0.1) 0%, transparent 50%), /* Matisse blue glow */
                radial-gradient(circle at 75% 75%, rgba(45, 85, 135, 0.1) 0%, transparent 50%); /* Darker Matisse blue glow */
            animation: pulse 15s infinite alternate;
            z-index: -1;
        }
        
        @keyframes pulse {
            0% { opacity: 0.5; }
            50% { opacity: 0.8; }
            100% { opacity: 0.5; }
        }
        
        /* Improve text readability */
        .stApp {
            color: white !important;
        }
        
        /* Make cards and inputs more visible with semi-transparent background */
        .stTextInput, .stSelectbox, .stMultiselect, .stTextArea, .stButton button {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border-radius: 5px;
            padding: 5px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Style the sidebar */
        .sidebar .sidebar-content {
            background-color: rgba(0, 0, 0, 0.4) !important;
        }
        
        /* Style the chat container */
        div[data-testid="stChatMessage"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Style the header */
        h1, h2, h3 {
            color: #4b8cde !important; /* Matisse blue color */
            text-shadow: 0 0 10px rgba(75, 140, 222, 0.4);
        }
        
        /* Style the tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(0, 0, 0, 0.4);
            border-radius: 5px 5px 0 0;
            padding: 10px 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(15, 77, 212, 0.2) !important; /* Matisse blue with opacity */
            border-bottom: 2px solid #0f4dd4 !important; /* Matisse blue border */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the background
add_bg_gradient()

# Initialize the model
llm_engine = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.5,
    max_new_tokens=20000,
    task="text-generation"
)

# Sidebar Settings
with st.sidebar:
    st.header("Chatbot Settings")
    model_name = st.selectbox("Select Model", ["gpt-4o", "deepseek-r1:3b"], index=0)
    st.subheader("Chat Settings")
    level = st.selectbox("Chat Level", ["Beginner", "Intermediate", "Expert"], index=0)
    st.subheader("Task Selection")
    task = st.selectbox("Select Task", [
        "General Code Explanation",
        "Optimization Suggestions",
        "Bug Detection and Fixes",
        "Security Vulnerability Analysis",
        "Best Practices Recommendations"
    ], index=0)

# Title and Description


if (model_name== "gpt-4o"):
    llm_engine=ChatOpenAI(model=model_name,temperature=0.65)
elif (model_name=="deepseek-r1:3b"):
    llm_engine = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-R1",  
    temperature=0.5,
    max_new_tokens=20000,
    task="text-generation"
)    
st.title("Smart Documentation and Code Explanation")
st.caption("AI system that automatically generates clear, detailed documentation and code explanations from existing codebases. The solution should bridge the gap between complex code and natural language, making technical details more accessible to developers of all levels.")

# Level-specific instructions dictionary
level_instructions = {
    "Beginner": {
        "specific_instructions": """
        - Break down every line of code into simple, plain English
        - Define all technical terms, even basic programming concepts
        - Use analogies and real-world examples for complex concepts
        - Include step-by-step explanations with comments
        - Explain the 'why' behind each operation
        - Add warnings about common mistakes and pitfalls
        """,
        "additional_guidelines": """
        - Use simple vocabulary and avoid technical jargon where possible
        - Include visual explanations or ASCII diagrams when helpful
        - Provide multiple examples starting with the simplest use case
        - Link to basic tutorials and learning resources
        """
    },
    "Intermediate": {
        "specific_instructions": """
        - Focus on key functionality and important implementation details
        - Explain advanced concepts and design patterns when relevant
        - Discuss error handling and edge cases
        """,
        "additional_guidelines": """
        - Balance between detailed explanations and concise documentation
        - Include practical examples with real-world scenarios
        - Reference related concepts and alternative approaches
        """
    },
    "Expert": {
        "specific_instructions": """
        - Focus on high-level architecture and design decisions
        - Highlight complex edge cases and performance considerations
        - Document any non-obvious implementations or optimizations
        - Include relevant algorithm complexity analysis
        - Focus on scalability and maintenance aspects
        """,
        "additional_guidelines": """
        - Keep explanations concise and technically precise
        - Reference advanced documentation and relevant papers
        - Discuss system-level implications and integration considerations
        - Include benchmark comparisons when relevant
        - Focus on optimization opportunities and trade-offs
        """
    }
}

prompt_templates = {
    "General Code Explanation": """
        Dont provide any Revised Code for this u just need to generate text
        Stick to explaining the code given
        You are a code documentation expert. Explain the given code in detail to the {level} level programmers:
        
        {specific_instructions}

        Base requirements for all levels:
        - Maintain technical accuracy
        - Focus on clarity and relevance
        - Suggest optimization improvements
        - Provide a revised code sample at the end

        Additional guidelines:
        {additional_guidelines}

Please generate documentation that matches this expertise level and in a detailed manner .
    """,
    "Optimization Suggestions": """
        Analyze the provided code and suggest optimizations for {level} level programmers:
        - Performance improvements
        - Memory management
        - Readability enhancements
        - Highlight best practices and efficiency considerations
        - Focus on optimization opportunities and trade-offs
        -provide the final optimized code
    Please generate documentation that matches this expertise level and in a detailed manner
    """,
    "Bug Detection and Fixes": """
        Review the given code and identify potential bugs for {level} level programmers. Provide:
        - Explanation of the issues
        - Suggested fixes with corrected code snippets
        - Focus on Tradeoffs when chaning the code
    Please generate documentation that matches this expertise level and in a detailed manner
    """,
    "Security Vulnerability Analysis": """
        Examine the provided code for security vulnerabilities for {level} level programmers. Cover:
        - Possible security threats
        - Best practices for secure coding
        - Recommendations to fix vulnerabilities
    Please generate documentation that matches this expertise level and in a detailed manner
    """,
    "Best Practices Recommendations": """
        Evaluate the code against software engineering best practices for {level} level programmers. Offer:
        - Clean coding guidelines
        - Maintainability improvements
    Please generate documentation that matches this expertise level and in a detailed manner
    """
}

if task=="General Code Explanation":
    system_prompt = SystemMessagePromptTemplate.from_template(
    prompt_templates[task],
    partial_variables={
        "level": level,
        "specific_instructions": level_instructions[level]["specific_instructions"],
        "additional_guidelines": level_instructions[level]["additional_guidelines"]
    })
else:
    system_prompt = SystemMessagePromptTemplate.from_template(
    prompt_templates[task],
    partial_variables={
        "level": level,
    })

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role":"ai","content":"Hello, I am Nebby, your AI coding assistant. How can I help you today?"}]
if 'user_query' not in st.session_state:
    st.session_state.user_query = None

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

# Add tabs to separate content
ai_tab, viz_tab ,usg_tab= st.tabs(["AI Response", "Code Visualization","Code Performance Analyzer"])

# Save the user query in session state
st.session_state.user_query = user_query
# First, save the visualization data in session state
if 'visualization_data' not in st.session_state:
    st.session_state.visualization_data = None   

# AI Response Tab
with ai_tab:
    if user_query:
        st.session_state.message_log.append({"role": "user", "content": user_query})
    
        with st.spinner("Thinking..."):
            prompt_chain = build_prompt_chain()
            ai_response = generate_ai_response(prompt_chain=prompt_chain)
        
        st.session_state.message_log.append({"role": "ai", "content": ai_response})
        st.write(ai_response)

# Visualization Tab
with viz_tab:
    code=st.text_area("Enter your message here...") 
    if st.button("Generate Flowchart"):
        streamlit_ui(code)

# Visualization Tab
with usg_tab:
    optimized_code = st.text_area("Enter your optimized Python code here:", height=200)
    unoptimized_code = st.text_area("Enter your unoptimized Python code here:", height=200)
    if st.button("Run Codes"):
        visualisation(optimized_code,unoptimized_code)
        
            



        
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import(
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from visual import streamlit_ui
from codeusage import  visualisation


#intialise the env

from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
import os


# Streamlit Page Config
st.set_page_config(
    page_title="Smart Documentation & Code Explanation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS-only background with Matisse blue styling
def add_bg_gradient():
    st.markdown(
        """
        <style>
        /* Create a dark gradient background */
        .stApp {
            background: linear-gradient(to bottom right, #1e1e2f, #000000);
        }
        
        /* Add a subtle animated pattern overlay */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(circle at 25% 25%, rgba(75, 140, 222, 0.1) 0%, transparent 50%), /* Matisse blue glow */
                radial-gradient(circle at 75% 75%, rgba(45, 85, 135, 0.1) 0%, transparent 50%); /* Darker Matisse blue glow */
            animation: pulse 15s infinite alternate;
            z-index: -1;
        }
        
        @keyframes pulse {
            0% { opacity: 0.5; }
            50% { opacity: 0.8; }
            100% { opacity: 0.5; }
        }
        
        /* Improve text readability */
        .stApp {
            color: white !important;
        }
        
        /* Make cards and inputs more visible with semi-transparent background */
        .stTextInput, .stSelectbox, .stMultiselect, .stTextArea, .stButton button {
            background-color: rgba(255, 255, 255, 0.1) !important;
            border-radius: 5px;
            padding: 5px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Style the sidebar */
        .sidebar .sidebar-content {
            background-color: rgba(0, 0, 0, 0.4) !important;
        }
        
        /* Style the chat container */
        div[data-testid="stChatMessage"] {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Style the header */
        h1, h2, h3 {
            color: #4b8cde !important; /* Matisse blue color */
            text-shadow: 0 0 10px rgba(75, 140, 222, 0.4);
        }
        
        /* Style the tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: rgba(0, 0, 0, 0.4);
            border-radius: 5px 5px 0 0;
            padding: 10px 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(15, 77, 212, 0.2) !important; /* Matisse blue with opacity */
            border-bottom: 2px solid #0f4dd4 !important; /* Matisse blue border */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the background
add_bg_gradient()

# Initialize the model
llm_engine = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.5,
    max_new_tokens=20000,
    task="text-generation"
)

# Sidebar Settings
with st.sidebar:
    st.header("Chatbot Settings")
    model_name = st.selectbox("Select Model", ["gpt-4o", "deepseek-r1:3b"], index=0)
    st.subheader("Chat Settings")
    level = st.selectbox("Chat Level", ["Beginner", "Intermediate", "Expert"], index=0)
    st.subheader("Task Selection")
    task = st.selectbox("Select Task", [
        "General Code Explanation",
        "Optimization Suggestions",
        "Bug Detection and Fixes",
        "Security Vulnerability Analysis",
        "Best Practices Recommendations"
    ], index=0)

# Title and Description


if (model_name== "gpt-4o"):
    llm_engine=ChatOpenAI(model=model_name,temperature=0.65)
elif (model_name=="deepseek-r1:3b"):
    llm_engine = HuggingFaceEndpoint(
    model="deepseek-ai/DeepSeek-R1",  
    temperature=0.5,
    max_new_tokens=20000,
    task="text-generation"
)    
st.title("Smart Documentation and Code Explanation")
st.caption("AI system that automatically generates clear, detailed documentation and code explanations from existing codebases. The solution should bridge the gap between complex code and natural language, making technical details more accessible to developers of all levels.")

# Level-specific instructions dictionary
level_instructions = {
    "Beginner": {
        "specific_instructions": """
        - Break down every line of code into simple, plain English
        - Define all technical terms, even basic programming concepts
        - Use analogies and real-world examples for complex concepts
        - Include step-by-step explanations with comments
        - Explain the 'why' behind each operation
        - Add warnings about common mistakes and pitfalls
        """,
        "additional_guidelines": """
        - Use simple vocabulary and avoid technical jargon where possible
        - Include visual explanations or ASCII diagrams when helpful
        - Provide multiple examples starting with the simplest use case
        - Link to basic tutorials and learning resources
        """
    },
    "Intermediate": {
        "specific_instructions": """
        - Focus on key functionality and important implementation details
        - Explain advanced concepts and design patterns when relevant
        - Discuss error handling and edge cases
        """,
        "additional_guidelines": """
        - Balance between detailed explanations and concise documentation
        - Include practical examples with real-world scenarios
        - Reference related concepts and alternative approaches
        """
    },
    "Expert": {
        "specific_instructions": """
        - Focus on high-level architecture and design decisions
        - Highlight complex edge cases and performance considerations
        - Document any non-obvious implementations or optimizations
        - Include relevant algorithm complexity analysis
        - Focus on scalability and maintenance aspects
        """,
        "additional_guidelines": """
        - Keep explanations concise and technically precise
        - Reference advanced documentation and relevant papers
        - Discuss system-level implications and integration considerations
        - Include benchmark comparisons when relevant
        - Focus on optimization opportunities and trade-offs
        """
    }
}

prompt_templates = {
    "General Code Explanation": """
        Dont provide any Revised Code for this u just need to generate text
        Stick to explaining the code given
        You are a code documentation expert. Explain the given code in detail to the {level} level programmers:
        
        {specific_instructions}

        Base requirements for all levels:
        - Maintain technical accuracy
        - Focus on clarity and relevance
        - Suggest optimization improvements
        - Provide a revised code sample at the end

        Additional guidelines:
        {additional_guidelines}

Please generate documentation that matches this expertise level and in a detailed manner .
    """,
    "Optimization Suggestions": """
        Analyze the provided code and suggest optimizations for {level} level programmers:
        - Performance improvements
        - Memory management
        - Readability enhancements
        - Highlight best practices and efficiency considerations
        - Focus on optimization opportunities and trade-offs
        -provide the final optimized code
    Please generate documentation that matches this expertise level and in a detailed manner
    """,
    "Bug Detection and Fixes": """
        Review the given code and identify potential bugs for {level} level programmers. Provide:
        - Explanation of the issues
        - Suggested fixes with corrected code snippets
        - Focus on Tradeoffs when chaning the code
    Please generate documentation that matches this expertise level and in a detailed manner
    """,
    "Security Vulnerability Analysis": """
        Examine the provided code for security vulnerabilities for {level} level programmers. Cover:
        - Possible security threats
        - Best practices for secure coding
        - Recommendations to fix vulnerabilities
    Please generate documentation that matches this expertise level and in a detailed manner
    """,
    "Best Practices Recommendations": """
        Evaluate the code against software engineering best practices for {level} level programmers. Offer:
        - Clean coding guidelines
        - Maintainability improvements
    Please generate documentation that matches this expertise level and in a detailed manner
    """
}

if task=="General Code Explanation":
    system_prompt = SystemMessagePromptTemplate.from_template(
    prompt_templates[task],
    partial_variables={
        "level": level,
        "specific_instructions": level_instructions[level]["specific_instructions"],
        "additional_guidelines": level_instructions[level]["additional_guidelines"]
    })
else:
    system_prompt = SystemMessagePromptTemplate.from_template(
    prompt_templates[task],
    partial_variables={
        "level": level,
    })

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role":"ai","content":"Hello, I am Nebby, your AI coding assistant. How can I help you today?"}]
if 'user_query' not in st.session_state:
    st.session_state.user_query = None

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

# Add tabs to separate content
ai_tab, viz_tab ,usg_tab= st.tabs(["AI Response", "Code Visualization","Code Performance Analyzer"])

# Save the user query in session state
st.session_state.user_query = user_query
# First, save the visualization data in session state
if 'visualization_data' not in st.session_state:
    st.session_state.visualization_data = None   

# AI Response Tab
with ai_tab:
    if user_query:
        st.session_state.message_log.append({"role": "user", "content": user_query})
    
        with st.spinner("Thinking..."):
            prompt_chain = build_prompt_chain()
            ai_response = generate_ai_response(prompt_chain=prompt_chain)
        
        st.session_state.message_log.append({"role": "ai", "content": ai_response})
        st.write(ai_response)

# Visualization Tab
with viz_tab:
    code=st.text_area("Enter your message here...") 
    if st.button("Generate Flowchart"):
        streamlit_ui(code)

# Visualization Tab
with usg_tab:
    optimized_code = st.text_area("Enter your optimized Python code here:", height=200)
    unoptimized_code = st.text_area("Enter your unoptimized Python code here:", height=200)
    if st.button("Run Codes"):
        visualisation(optimized_code,unoptimized_code)
