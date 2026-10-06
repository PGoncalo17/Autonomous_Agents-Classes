import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()

base_url = os.environ["OLLAMA_BASE_URL"]
api_key = os.environ["OLLAMA_API_KEY"]
model = os.environ["OLLAMA_MODEL"]
llm = ChatOllama(model=model, 
                base_url=base_url,
                temperature = 0,
                # top_k = 15,
                # top_p = 1,
                # min_p = 0.3,
                # repeat_penalty = 1.5,
                # repeat_last_n = 64,
                client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}}
                )

web_search_tool = TavilySearch(max_results=3)
tools = [web_search_tool]

llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content=(
        """You are a helpful assistant with access to a web search tool.
        Only search the web if you need up-to-date facts or external information.
        For simple greetings or general conversation, respond naturally without searching."""
    )
)



def chatbotWithTools(state: MessagesState):
    messages = state["messages"]
    if not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(MessagesState)

graph.add_node("chatbotWithTools", chatbotWithTools)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chatbotWithTools")
graph.add_conditional_edges("chatbotWithTools", tools_condition)
graph.add_edge("tools", "chatbotWithTools")    
app = graph.compile()


if __name__ == "__main__":
    print("ChatBot: Hello! I'm ChatBot, what can I do for you today?\n")

    conversation_history = []

    while(True):
        user_input = input("You: ")
        if(user_input.lower() == "exit"):
            print("ChatBot: Goodbye. It was nice chatting with you!")
            break

        conversation_history.append(HumanMessage(content=user_input))

        response_state = app.invoke({"messages": conversation_history})
        conversation_history = response_state["messages"]

        latest_response = conversation_history[-1]
        print(f"ChatBot:  {latest_response.content}\n")
