from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os

from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

import json
from typing import Literal
from langgraph.graph import END

load_dotenv()

base_url = os.environ["OLLAMA_BASE_URL"]
api_key = os.environ["OLLAMA_API_KEY"]
model = os.environ["OLLAMA_MODEL"]
llm = ChatOllama(model="qwen3:8b",
    base_url=base_url,
    temperature=0,
    seed = 33
)
search = TavilySearch(max_results=3)

class State(MessagesState):
    classification: str
    query: str
    search_results: str

def classifier(state: State):
    question = state["messages"][-1].content

    system = SystemMessage(content = """
    You are a web search classifier.

    Your task is to decide wether the user's question requiers web search.

    return WEB_SEARCH when the question requires:
    - Current or recent information.
    - Updated facts.
    - Information that may have changed.
    - External verification

    Return NO_WEB_SEARCH when the question requires:
    - general knowledge.
    - Explanation of concepts.
    - Mathematics.
    - Creative writting.
    - Casual conversation.
    - Information already provided by the user.

    If you aree unsure whether the information is up to date, choose WEB_SEARCH.

    return only a valid JSON object with two fields:
    [
        "classification": "WEB_SEARCH or NO_WEB_SEARCH",
        "query": "search query or empty string"
    ]

    If the classification is NO_WEB_SEARCH, the query must be an empty string.

    """)

    response = llm.invoke([
        system,
        HumanMessage(content = question)]
    )

    data = json.loads(response.content.strip())
    classification = data["classification"].strip().upper()
    query = data["query"].strip()

    if classification not in ["WEB_SEARCH", "NO_WEB_SEARCH"]:
        classification = "WEB_SEARCH"

    if classification == "NO_WEB_SEARCH":
        query = ""

    if classification == "WEB_SEARCH" and not query:
        query = question

    print("\nClassifier", classification)
    print("search query", query)

    return{
        "classification": classification,
        "query": query,
        "search_results": ""
    }        

#======================================
# router
#======================================

def router(state: State) -> Literal["web_search", "model"]:
    classification = state["classification"]
    if classification == "WEB_SEARCH":
        print("ROUTER: going io web search")
        return "web_search"
    print("ROUTER: going directly to model")
    return "model"

#======================================
# web search node
#======================================

def web_search(state: State):
    query = state["query"]
    print("\nWEB_SEARCH", query)

    results = search.invoke({
        "query": query
    })

    return{
        "search results": json.dumps(
        results,
        ensure_ascii=False
        )
    }

#=======================================
# Model node
#=======================================

def model(state:State):

    question = next(
        message.content
        for message in reversed(state["messages"])
        if isinstance(message, HumanMessage)
    )

    system_prompt = """

    You are a helpful and reiable AI assistant.
    Answer in the same language as the user writes in.

    If web search results are available, use them to answer the user's question.

    If no web search results are available, answer using your existing knowledge.

    Do not invent facts.
    """

    if state.get("search_results"):
        system_prompt += f"""
        WEB SEARCH RESULTS:
        {state["search_results"]}
    """
    response = llm.invoke([
        SystemMessage(content = system_prompt),
        *state["messages"]
    ])

    return {
        "messages": [response],
        "search_results": ""
    }


#=======================================
# graph
#=======================================
graph = StateGraph(MessagesState)

graph.add_node("classifier", classifier)
graph.add_node("web_search", web_search)
graph.add_node("model", model)

graph.add_edge(START, "classifier")
graph.add_conditional_edges("classifier", router)
graph.add_edge("web_search", "model")

graph.add_edge("model", END)
app = graph.compile()


if __name__ == "__main__":
    print("to quit enter 'exit'.\n")

    messages = []
    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break
        messages.append(
            HumanMessage(content = question)
        )
        result = app.invoke({
            "messages": messages,
            "search_results": ""
            })
        messages = result["messages"]


        print ("Agent:", result['messages'][-1].content)