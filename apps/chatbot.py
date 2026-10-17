from dotenv import load_dotenv
import os
load_dotenv()  
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
llm=ChatGroq(model="openai/gpt-oss-20b",temperature=0.7)
st.title("🤖 Chatbot - Your true companion!")
st.markdown("You can talk to me anytime and I will always be there to listen and respond.  Let's chat!")
if "messages" not in st.session_state:
    st.session_state.messages=[]
for message in st.session_state.messages:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content) 
query = st.chat_input("Ask me anything:")
if query:
    st.session_state.messages.append({"role":"user","content":query})   
    st.chat_message("user").markdown(query)
    history=[]
    for msg in st.session_state.messages:
        if msg["role"]=="user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    res=llm.invoke(history)
    st.chat_message("assistant").markdown(res.content)
    st.session_state.messages.append({"role":"assistant","content":res.content})    
