import json
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# CHANGE THIS TO POINT TO YOUR CLONED REPO
# ==========================================
repo_path = "../flask-login" 

print(f"Scanning entire repository at: {repo_path}...")

# 1. Dynamically load EVERY file in the repository recursively
# We use TextLoader as the engine to read files as plain text strings.
loader = DirectoryLoader(
    repo_path, 
    glob="**/*", 
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    silent_errors=True # Skips binary files like images or compiled bytes safely
)

docs = loader.load()
print(f"Successfully scanned and loaded {len(docs)} files from the repository.")

# 2. Split into text chunks (Standard Vector RAG practice)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} total text chunks across the whole codebase.")

# 3. Create the Flat Vector Store (Baseline B1)
openrouter_key = os.environ.get("OPENROUTER_API_KEY")

if not openrouter_key:
    print("WARNING: You forgot to set your OPENROUTER_API_KEY in the terminal!")
    exit()

print("Setting up free OpenRouter embeddings...")
embeddings = OpenAIEmbeddings(
    api_key=openrouter_key,
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    check_embedding_ctx_length=False, 
    encoding_format="float"
)

print("Embedding chunks and building vector database...")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 

# 4. Set up the Prompt and LLM
print("Setting up free OpenRouter chat model...")
llm = ChatOpenAI(
    api_key=openrouter_key,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-oss-120b:free",
    temperature=0
)
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}
""")
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 5. Your Full 10-Question Ground Truth Dataset
questions = [
    "The documentation explains how to set a 'Custom Login View' that users are redirected to when they try to access a protected page. Which specific Python class and attribute implements this configuration?",
    "According to the documentation, how should a custom user class behave if an invalid ID is passed to the `user_loader` callback? Which file handles removing that invalid ID from the active session if this scenario occurs?",
    "If a developer wants to authenticate users via an API token passed through a header parameter instead of using traditional cookies, what specific callback decorator must they define? Provide a brief code reference showing how it accesses the request data.",
    "Trace the execution flow when the `@login_required` decorator is triggered by an unauthenticated user. Which function is called to handle the unauthorized request?",
    "When a login session protection check fails in `strong` mode for a non-permanent session, what exact cleanup actions occur on the session dictionary, and what specific Flask signal is triggered?",
    "Trace the data flow of `encode_cookie()`. What standard hashing algorithm and helper methodology are used to protect the user ID payload from being altered by client-side cookie tampering?",
    "When a user triggers the `logout_user()` function, what exact keys are popped out of the Flask session dictionary, and how does the application signal to the browser that the 'remember me' cookie must be deleted?",
    "What is the primary mechanism `flask-login` uses to remember a user's session across different requests?",
    "How does Flask-Login conceptually distinguish a 'fresh' login session from a 'non-fresh' session? What utility decorator protects sensitive actions like updating password or personal settings?",
    "From a high-level architectural perspective, when Flask-Login needs to resolve who the current user is for an incoming request, what are the three distinct fallback mechanisms it checks, and in what order?"
]

# 6. Run the Test Cycle
results = []
print("Starting baseline evaluation execution query cycle...")

for i, q in enumerate(questions, 1):
    print(f"Running Question {i}/10...")
    response = retrieval_chain.invoke({"input": q})
    
    # Track the exact text fragments used to pass into RAGAS later
    retrieved_contexts = [doc.page_content for doc in response["context"]]
    
    results.append({
        "question": q,
        "answer": response["answer"],
        "contexts": retrieved_contexts
    })

# 7. Save the raw output for your RAGAS script
output_file = "baseline_vector_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
    
print(f"Initialization complete! Baseline raw results saved to {output_file}.")