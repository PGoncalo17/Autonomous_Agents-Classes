
"""
Basic LLM-based chatbot with LangGraph - CHATBOT COM HumanMessage
"""
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import (HumanMessage, AIMessage)
from langgraph.checkpoint.sqlite import SqliteSaver



load_dotenv()

# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
model="llama3.1:latest",
base_url="https://agentesautonomos.iscte-iul.pt",
temperature=0,
seed=33
)

# ============================================================
# CHATBOT NODE
# ============================================================

def chatbot(state: MessagesState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {
    "messages": [response]
    }

# ============================================================
# USERS - FIXED LIST
# ============================================================

users = {
    "1": "marta",
    "2": "joao"
}

# ============================================================
# SELECT USER
# ============================================================

print("Select a user: ")
print("1: Marta")
print("2: João")

while True:
    user_choice = input("User: ")
    
    if user_choice in users:
        break
    print("Invalid user choice. Please selct 1 or 2")

selected_user = users[user_choice]
print(f"\nSelected user: {selected_user}\n")

# ============================================================
# GRAPH
# ============================================================

graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START,"chatbot")
graph.add_edge("chatbot",END)

with SqliteSaver.from_conn_string("conversation_history.db") as memory:
    app = graph.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": selected_user}}


# ============================================================
# CHAT
# ============================================================
    print("Chatbot.To quit enter 'exit'.")
    messages = []


    while True:
        question = input("You: ")
        if question == "exit":
            break
        messages.append(HumanMessage(content=question))
        result = app.invoke({"messages": HumanMessage(content=question)}, config=config )
        messages= result["messages"]
        print("Agent:",result["messages"][-1].content)
        print("Message in state: ", len(result["messages"]))
        print()