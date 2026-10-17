from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st


db=SQLDatabase.from_uri("sqlite:///my_database.db")


db.run("""
    CREATE TABLE IF NOT EXISTS tasks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT check(status in ('pending', 'in_progress', 'completed')) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        
    );
""")

model=ChatGroq(model="openai/gpt-oss-20b")
toolkit=SQLDatabaseToolkit(db=db, llm=model)
tools=toolkit.get_tools() 
memory=InMemorySaver()

system_prompt = """
You are a task management assistant that interacts with a SQLite database containing a 'tasks' table.

TASK RULES:
1. Limit SELECT queries to recent results.
2. After INSERT, UPDATE, or DELETE operations, verify the change with a SELECT query.
3. If the user requests a list of tasks, present the output in a clean table format.
4. Never invent data that does not exist in the database.
5. Use only the available table and columns.
6. If a task is not found, clearly inform the user.
7. Ask for confirmation before deleting all tasks.

CRUD OPERATIONS:

CREATE:
INSERT INTO tasks(title, description, status)

READ:
SELECT * FROM tasks
WHERE ...
ORDER BY created_at DESC
LIMIT 10

UPDATE:
UPDATE tasks
SET title=?, description=?, status=?
WHERE id=?

DELETE:
DELETE FROM tasks
WHERE id=?

TABLE SCHEMA:

tasks(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT CHECK(status IN (
        'pending',
        'in_progress',
        'completed'
    )) DEFAULT 'pending'
)

STATUS VALUES:
- pending
- in_progress
- completed

AGENT BEHAVIOR:
- Understand the user's request.
- Generate valid SQLite queries.
- Execute the query using available tools.
- Return concise and accurate responses.
- For task listings, display data in tabular format.
- For counts, return only the relevant number and explanation.
- Never generate DROP TABLE or destructive schema changes.
"""

@st.cache_resource
def get_agent():
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver()
    )
    return agent
agent=get_agent()

st.subheader("TaskBot - Your Task Management Assistant")
if "messages" not in st.session_state:
       st.session_state.messages=[]

for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Ask me anything about your tasks...")
if prompt: 
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role":"user" , "content":prompt})
    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
                    response = agent.invoke(
                        {"messages":[{"role":"user" , "content":prompt}]},
                        {"configurable":{"thread_id": "1"}}
                    )
                    result=response["messages"][-1].content
                    st.markdown(result)
                    st.session_state.messages.append({"role":"ai" , "content":result})