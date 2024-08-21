import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Setting up the page configuration with title and icon
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")

# Setting up the title of the app
st.title("🦜 LangChain: Chat with SQL DB")

# Retrieve the Groq API key from the .env file
api_key = os.getenv("GROQ_API_KEY")

# Check if the API key is provided
if not api_key:
    st.error("Please set the Groq API key in the .env file.")
    st.stop()

# Initialize the Groq LLM
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

# Function to configure SQLite database
@st.cache_resource(ttl="2h")
def configure_db():
    dbfilepath = (Path(__file__).parent / "analytics_db").absolute()
    creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
    return SQLDatabase(create_engine("sqlite:///", creator=creator))

# Configure DB
db = configure_db()

# SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)  # Directly pass llm object

# Creating an agent with SQL DB and Groq LLM
agent = create_sql_agent(
    llm=llm,  # Pass llm directly
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Session state for messages (clear button available)
if "messages" not in st.session_state or st.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input for user query
user_query = st.chat_input(placeholder="Ask anything from the database")

# If user query is submitted
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Generate response from agent
    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(user_query, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Ensure the response is in tabular format
            if isinstance(response, list):
                if all(isinstance(i, tuple) for i in response) and len(response) > 0:
                    # Assuming the first tuple contains the headers
                    headers = [f"Column {i+1}" for i in range(len(response[0]))]
                    df = pd.DataFrame(response, columns=headers)
                    st.dataframe(df.style.set_properties(**{'color': 'white', 'background-color': 'black'}))
                else:
                    st.write("The response is not in tabular format.")
            else:
                st.write("The response is not in tabular format.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
