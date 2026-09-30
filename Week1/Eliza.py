import random

sad_keyWord = ["sad", "depressed", "unhappy", "cried", "cry"]
happy_keyWord = ["happy", "good", "joyfull", "thrilled"]
goodbye_keyWord = ["goodbye", "bye", "see you later", "take care"]

sad_answer = ["What happened?", "Everything's good?", "Don't worry"]
happy_answer = ["Good!", "I'm happy for you", "You are amazing"]
goodbye_answer = ["Goodbye", "Take care", "Thank you for sharing with me!"]

random_seed = 71
random.seed(random_seed)


while True:
    text = input("I'm Eliza: ")
    if any(keyword in text.lower() for keyword in sad_keyWord):
        print(random.choice(sad_answer))
    elif any(keyword in text.lower() for keyword in happy_keyWord):
        print(random.choice(happy_answer))
    elif any(keyword in text.lower() for keyword in goodbye_keyWord):
        print(random.choice(goodbye_answer))
        break    
    elif any(text.lower()):
        print("Type a different text please. That is beyound my capacities, I'm sorry! :(") 
         