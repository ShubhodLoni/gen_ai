from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated

class ChatState(BaseModel):
    messages: Annotated[list, add_messages]

llm = ChatGroq(model="openai/gpt-oss-20b")
def ChatBot(state: ChatState) ->  ChatState:
    res = llm.invoke(state.messages)
    state.messages = [res]
    return state

memory = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chatbot", ChatBot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

graph = graph.compile(checkpointer=memory)

#from IPython.display import Image
#Image(graph.get_graph().draw_mermaid_png())

config = {"configurable":{"thread_id":"chatbot_thread"}}
while True:
    query = input("User: ")
    if query.lower() in ["exit", "quit"]:
        print("Thank you for using the chatbot. Goodbye!")
        break
    res = graph.invoke({"messages":[{"role":"user","content":query}]},config)
    print(f"Chatbot: {res['messages'][-1].content}")