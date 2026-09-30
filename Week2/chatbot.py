from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import os

load_dotenv()

base_url = os.environ["OLLAMA_BASE_URL"]
api_key = os.environ["OLLAMA_API_KEY"]
model = os.environ["OLLAMA_MODEL"]
llm = ChatOllama(model=model, 
                base_url=base_url,
                temperature = 1.0,
                top_k = 15,
                top_p = 1,
                min_p = 0.3,
                repeat_penalty = 1.5,
                repeat_last_n = 64,
                client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}}
                )


def chatbot(user_input):
    llm_output = llm.invoke(user_input)
    return llm_output.content

    

print("ChatBot: Hello! I'm chatBot, what can I do for you today?\n")
while(True):
    user_input = input("You: ")
    if(user_input.lower() == "exit"):
        print("ChatBot: Goodbye. It was nice chatting with you!")
        break
    print(f"ChatBot:  {chatbot(user_input)}\n")

