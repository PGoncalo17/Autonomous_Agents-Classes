from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import os
from langgraph.graph import START, END, StateGraph, MessagesState
from typing import Annotated, TypedDict

load_dotenv()

base_url = os.environ["OLLAMA_BASE_URL"]
api_key = os.environ["OLLAMA_API_KEY"]
model = os.environ["OLLAMA_MODEL"]
llm = ChatOllama(model=model, 
                base_url=base_url,
                temperature = 0.3,
                top_k = 15,
                top_p = 1,
                min_p = 0.3,
                repeat_penalty = 1.5,
                repeat_last_n = 64,
                client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}}
                )

def add(a, b):
    return a + b

class State(TypedDict):
    messages: Annotated[list[str], add]



def chatbotWithState(state: State):
    messages = state["messages"]
    llm_output = llm.invoke(messages)
    return {"messages": [llm_output.content]}


graph = StateGraph(State)
graph.add_node("chatbotWithState", chatbotWithState)
graph.add_edge(START, "chatbotWithState")
graph.add_edge("chatbotWithState", END)    
app = graph.compile()


print("ChatBot: Hello! I'm ChatBot, what can I do for you today?\n")
while(True):
    user_input = input("You: ")
    user_input_dic = {"messages": [user_input]}
    if(user_input.lower() == "exit"):
        print("ChatBot: Goodbye. It was nice chatting with you!")
        break
    response = app.invoke(user_input_dic)
    print(f"ChatBot:  {response['messages'][-1]}\n")
