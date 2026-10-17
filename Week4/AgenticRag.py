import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

base_url = os.environ["OLLAMA_BASE_URL"]
api_key = os.environ["OLLAMA_API_KEY"]
model = os.environ["OLLAMA_MODEL"]

llm = ChatOllama(
    model=model,
    base_url=base_url,
    temperature=0,
    client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}}
)

embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5", 
    encode_kwargs={"normalize_embeddings": True}
)

persist_directory = "./chroma_db"

if not os.path.exists(persist_directory):
    doc_path = "rag_document.txt"
    if not os.path.exists(doc_path):
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Company XYZ was founded in 2021. Its main product is SmartRAG.")

    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
else:
    vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )

retriever_obj = vectorstore.as_retriever(search_kwargs={"k": 3})


@tool
def retriever(query: str) -> str:
    """Searches and retrieves relevant document chunks from the vector database based on the user's query."""
    docs = retriever_obj.invoke(query)
    
    context_chunks = []
    for doc in docs:
        sub_section = doc.metadata.get("sub_section", "")
        if sub_section:
            context_chunks.append(f"[{sub_section}] {doc.page_content}")
        else:
            context_chunks.append(doc.page_content)
            
    return "\n\n".join(context_chunks)


tools = [retriever]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are an intelligent assistant with access to a knowledge retrieval tool.\n"
        "INSTRUCTIONS:\n"
        "1. Always check if you need to retrieve information from the internal knowledge base to answer the user accurately.\n"
        "2. If the user asks specific questions about domain facts, call the `retriever` tool with an appropriate query.\n"
        "3. Evaluate the retrieved information and construct your answer based on it.\n"
        "4. If the retrieved information is insufficient, state that you do not have enough information.\n"
        "5. For casual greetings or general conversational statements, respond directly without invoking tools."
    )
)


def call_model(state: MessagesState):
    messages = state["messages"]
    filtered_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    prompt_messages = [SYSTEM_PROMPT] + filtered_messages
    
    response = llm_with_tools.invoke(prompt_messages)
    return {"messages": [response]}


graph = StateGraph(MessagesState)

graph.add_node("model", call_model)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "model")
graph.add_conditional_edges("model", tools_condition)
graph.add_edge("tools", "model")

app = graph.compile()


if __name__ == "__main__":
    print("\nChatBot: Hi! How can I help you today?\n")

    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("ChatBot: Goodbye!")
            break

        conversation_history.append(HumanMessage(content=user_input))

        response_state = app.invoke({"messages": conversation_history})
        conversation_history = response_state["messages"]

        latest_response = conversation_history[-1]
        print(f"ChatBot: {latest_response.content}\n")