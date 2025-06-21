from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import asyncio
from fastapi.responses import StreamingResponse
import json



from langchain.globals import set_debug

set_debug(True)

# Set your Google API key here
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    topic: str
    question: str

class TopicRequest(BaseModel):
    topic: str

# Caches
retrievers = {}
vectorstores = {}
# Simple in-memory chat memory per topic
chat_memory = {}
MAX_MEMORY_TURNS = 5  # max number of previous Q/A pairs to keep


# Shared components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3,streaming=True)

async def create_qa_chain(topic: str) -> RetrievalQA:
    print(f"Loading Wikipedia docs for topic: {topic} ...")
    loader = WikipediaLoader(query=topic, load_max_docs=5, lang="en")
    docs = loader.load()
    # print(f"Loaded {len(docs)} documents")

    # for i, doc in enumerate(docs):
    #     print(f"\n--- Wikipedia Document {i + 1} ---")
    #     print(doc.page_content[:1000])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_chunks = splitter.split_documents(docs)

    # Reuse or create vectorstore
    if topic not in vectorstores:
        vectorstore = FAISS.from_documents(docs_chunks, embeddings)
        vectorstores[topic] = vectorstore
    else:
        vectorstores[topic].add_documents(docs_chunks)  # Just add new docs

    retriever = vectorstores[topic].as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever,verbose=True)

    return qa_chain

@app.post("/ask")
async def ask_question(data: QuestionRequest):
    topic = data.topic.lower().strip()
    question = data.question.strip()

    if topic not in retrievers:
        retrievers[topic] = await create_qa_chain(topic)

    qa_chain = retrievers[topic]

    # Prepare context from memory
    history = chat_memory.get(topic, [])
    # Join previous turns into a string
    context_str = ""
    for i, (q, a) in enumerate(history[-MAX_MEMORY_TURNS:]):
        context_str += f"User: {q}\nBot: {a}\n"
    # Append current question
    prompt_with_history = context_str + f"User: {question}\nBot:"

    try:
        # Pass prompt_with_history as the query so the QA chain sees context
        result = qa_chain.invoke({"query": prompt_with_history})
        answer = result["result"]
    except Exception as e:
        print(f"Error in QA chain invoke: {e}")
        answer = "Sorry, I couldn't fetch the answer at this time."

    # Update memory
    if topic not in chat_memory:
        chat_memory[topic] = []
    chat_memory[topic].append((question, answer))

    # Optional: trim old memory if too large
    if len(chat_memory[topic]) > MAX_MEMORY_TURNS:
        chat_memory[topic] = chat_memory[topic][-MAX_MEMORY_TURNS:]

    return {"answer": answer}


@app.post("/change_topic")
async def change_topic(request: TopicRequest):
    topic = request.topic.lower().strip()
    print(f"Reloading topic: {topic}")

    # Just recreate the retriever using the same vectorstore (if exists)
    retrievers[topic] = await create_qa_chain(topic)
    return {"message": f"Topic changed to '{topic}' and retriever updated."}  

