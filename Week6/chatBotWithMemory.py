
"""
Basic LLM-based chatbot with LangGraph - CHATBOT COM HumanMessage
"""
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import (HumanMessage, AIMessage)
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

#============================================================
#LLM
#============================================================

llm = ChatOllama(
model="llama3.1:latest",
base_url="https://agentesautonomos.iscte-iul.pt",
temperature=0,
seed=33
)

#============================================================
#CHATBOT NODE
#============================================================

def chatbot(state: MessagesState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {
    "messages": [response]
    }

#============================================================
#GRAPH
#============================================================

graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START,"chatbot")
graph.add_edge("chatbot",END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "week6_chat"}}


#============================================================
#CHAT
#============================================================
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