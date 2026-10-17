import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import START, END, StateGraph, MessagesState

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
    loader = TextLoader(doc_path)
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

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def naive_rag_node(state: MessagesState):
    messages = state["messages"]
    last_user_input = messages[-1].content
    
    docs = retriever.invoke(last_user_input)
    
    context_chunks = []
    for doc in docs:
        sub_section = doc.metadata.get("sub_section", "")
        if sub_section:
            context_chunks.append(f"[{sub_section}] {doc.page_content}")
        else:
            context_chunks.append(doc.page_content)
            
    context = "\n\n".join(context_chunks)
    
    SYSTEM_PROMPT = (
        "You are a strict and helpful virtual assistant.\n\n"
        "EXPLICIT INSTRUCTIONS:\n"
        "- Answer the user's questions based EXCLUSIVELY on the provided context below.\n"
        "- If the answer is not contained within the provided context, you must explicitly state that you do not know.\n"
        "- Do NOT use any prior knowledge or information outside of the provided context.\n\n"
        f"CONTEXT:\n{context}"
    )
    
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    prompt_messages = [system_message] + messages
    
    response = llm.invoke(prompt_messages)
    return {"messages": [response]}


graph = StateGraph(MessagesState)

graph.add_node("naive_rag_node", naive_rag_node)
graph.add_edge(START, "naive_rag_node")
graph.add_edge("naive_rag_node", END)

app = graph.compile()


if __name__ == "__main__":
    print("\nChatBot: Hi! What can I do for you today?\n")

    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("ChatBot: See you later!")
            break

        conversation_history.append(HumanMessage(content=user_input))

        response_state = app.invoke({"messages": conversation_history})
        conversation_history = response_state["messages"]

        latest_response = conversation_history[-1]
        print(f"ChatBot: {latest_response.content}\n")