#==========================================
# 1. Imports
#==========================================

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Literal



#=========================================
# 2. Configuration
#=========================================

load_dotenv()

llm = ChatOllama(
    model= "qwen3:8b",
    base_url= "https://agentesautonomos.iscte-iul.pt",
    temperature=0
)

#=========================================
# 3. Personal profile -Sofia
#=========================================



PROFILE = """
Name : Sofia
Age: 24
Lives: Porto, Portugal
Work as: Junior software enginneer at a tech startup
Currently: Finishing a Master's degree in computer science at universidade do Porto
Sport: yoga, (3x/week) and cycling in weekends
Favourite food: lasagna and italian - hayes olives and mushrooms
Has a dog called pipoca, a golden retriever, 2 years old
wants to learn french
Dream trip: Japan
Stresses about: balancing work and thesis deadline.
Supervisor says she is very creative but sometimes loses focus.
"""

#======================================
# 4. State - custom messagesstate
#======================================

class State(MessagesState):
    
    classification: str

#======================================
# 5. Classifier node
#======================================

def classifier(state: State):
    system = SystemMessage(f"""
    You are a message classifier.
    Read the user's message and classify it as exactly one of:
    - "Sport" - If the message is about exercices, sport, physical activity, energy or body.

    - "career" - If the message is about work, career, thesis, deadlines, studying or academic life.

    - "Wellbeing" - If the message is about emotions, stress, anxiety, personal life or needing support.

    Respond with only one word: Sport, Career or Wellbeing.
    Nothing else
    """)
    response = llm.invoke([system] + state["messages"])
    classification = response.content.strip().lower()
    return {"classification": classification}

#======================================
# 6. Router function
#======================================

def router(state: State) -> Literal["sport", "career", "wellbeing"]:
    classification = state["classification"].lower()

    if "sport" in classification:
        return "sport"

    elif "career" in classification: 
        return "career"

    return "wellbeing"
    
#=====================================
# 7. Specialized node - Sport
#=====================================

def sport(state: State):
    system = SystemMessage(content = f"""
    {PROFILE}

    You are Sofia's personal sport and wellness coach.
    You know her routine well - yoga 3 times a week and cycling on weekends.
    Give pratical, personalised advice based on her lifestyle.
    Always respond in the same language as Sofia writes in.
    """)

    response = llm.invoke([system] + state["messages"])

    return {"messages": [response]}

#=====================================
# 8. Specialized node - Career
#=====================================

def career(state: State):
    system = SystemMessage(content = f"""
    {PROFILE}

    You are Sofia's personal career and academic advisor.
    You know she is balancing a startup job with finishing her Master's thesis.
    Her supervisor says that she is creative but sometimes loses her focus - keep that in mind.
    Give concrete, pratical advices about productivity, deadlines and career decisions. 
    Always respond in the same language as Sofia writes in.
    """)

    response = llm.invoke([system] + state["messages"])

    return {"messages": [response]}

#=====================================
# 9. Specialized node - Wellbeing
#=====================================

def wellbeing(state: State):
    system = SystemMessage(content = f"""
    {PROFILE}

    You are Sofia's warm and empathetic personal assistant.
    Use what you know about her - Pipoca the dog, her love for Italy, her dream of going to Japan- to respond with genuine, personalised empathy.
    Do not give unsolicited advice.
    Listen first, validate her feelings.
    Always respond in the same language as Sofia writes in.
    """)

    response = llm.invoke([system] + state["messages"])

    return {"messages": [response]}

#=====================================
# 10. Build the graph
#=====================================

graph = StateGraph(State)

graph.add_node("classifier", classifier)
graph.add_node("sport", sport)
graph.add_node("career", career)
graph.add_node("wellbeing", wellbeing)

graph.add_edge(START, "classifier")

graph.add_conditional_edges("classifier", router)

graph.add_edge("sport", END)
graph.add_edge("career", END)
graph.add_edge("wellbeing", END)

app = graph.compile()

#=======================================
# 11. Chat loop - test the assistance
#=======================================

if __name__ == "__main__":
    print("Sofia's personal Assistant")
    print("to quit enter 'exit'.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        result = app.invoke({
            "messages": [HumanMessage(content = question)]
            })

        print (f"\nClassification: {result['classification']}")

        print(f"\n Assistant: {result['messages'][-1].content}\n")    
