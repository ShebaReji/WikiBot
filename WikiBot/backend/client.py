import requests

print("Welcome to the Wikipedia RAG Chatbot!")

topic = input("Enter a topic to begin: ").strip()

while True:
    question = input(f"Ask a question about '{topic}' (or type 'exit' / 'change topic'): ").strip()

    if question.lower() == "exit":
        break

    if question.lower() == "change topic":
        topic = input("Enter a new topic: ").strip()
        
        change_response = requests.post("http://127.0.0.1:8000/change_topic", json={"topic": topic})
        if change_response.ok:
            print(change_response.json().get("message"))
        else:
            print("Error changing topic:", change_response.text)
        continue

    response = requests.post("http://127.0.0.1:8000/ask", json={"topic": topic, "question": question})
    if response.ok:
        print("Answer:", response.json().get("answer"))
    else:
        print("Error from server:", response.text)
