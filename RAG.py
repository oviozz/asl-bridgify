from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.schema import HumanMessage, SystemMessage
import uuid
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# app = Flask(__name__)

documents = []
metadatas = []

# Load API keys from environment variables
API_KEY = os.getenv('GOOGLE_API_KEY')
CSE_ID = os.getenv('GOOGLE_CSE_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

loader = PyPDFLoader("American Sign Language The Easy Way (Stewart).pdf")
pages = loader.load_and_split()


with open("/Users/aahilali/Desktop/asl-bridgify/ASLtext.txt", "r", encoding="utf-8") as file:
    text = file.read()
print(text)

class TextLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        with open(self.filepath, "r") as file:
            return file.read()

class Document:
    def __init__(self, content, metadata=None):
        self.content = content
        self.metadata = metadata or {}

class TextSplitter:
    def __init__(self, delimiter="\n\n"):
        self.delimiter = delimiter

    def split(self, text):
        return text.split(self.delimiter)

# Example usage




# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=100,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )

# texts = text_splitter.create_documents([asl_text])
# print(texts)
# Function to call Google Search API
def google_search(query, num_results=10):
    url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        'key': 'AIzaSyB1KY4MHbkXucegKL0f0CIaXvvPQeFZP-0',
        'cx': '407779dc524074fb4',
        'q': query,
        'num': num_results
    }
    response = requests.get(url, params=params)
    results = response.json().get('items', [])
    return [result['snippet'] for result in results]

# Function to call Google Search API and return documents
def google_search1(query, num_results=10):
    url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        'key':'AIzaSyB1KY4MHbkXucegKL0f0CIaXvvPQeFZP-0',
        'cx': '407779dc524074fb4',
        'q': query,
        'num': num_results
    }
    response = requests.get(url, params=params)
    results = response.json().get('items', [])

    documents = []
    for result in results:
        page_content = result.get('snippet', '')
        title = result.get('title', '')
        link = result.get('link', '')

        document = Document(
            page_content=page_content,
            metadata={
                'title': title,
                'link': link
            }
        )
        documents.append(document)

    return documents

# Load ChatQA model and tokenizer
llm = ChatOpenAI(api_key="sk-proj-e55E248XEt1lI8Wi78pXT3BlbkFJlQxFwhqebN03i322y6hZ", temperature=0)

# Function to create embeddings and FAISS index
def create_faiss_index(strings, documents):
    embedding_model = OpenAIEmbeddings(openai_api_key="sk-proj-e55E248XEt1lI8Wi78pXT3BlbkFJlQxFwhqebN03i322y6hZ")
    document_embeddings = embedding_model.embed_documents(strings)

    dbIndex = FAISS.from_documents(documents, embedding_model)
    return dbIndex

def create_pdf_faiss_index(documents):
    embedding_model = OpenAIEmbeddings(openai_api_key="sk-proj-e55E248XEt1lI8Wi78pXT3BlbkFJlQxFwhqebN03i322y6hZ")
    dbIndex = FAISS.from_documents(documents, embedding_model)
    return dbIndex

# Main function to answer query using web search and ChatQA with system instructions
def answer_query_with_rag(user_query, system_instructions):
    # Step 1: Get web search results
    search_results = google_search1(user_query)
    search_results1 = google_search(user_query)

    documents.append(user_query)
    metadatas.append({"role": "user", "id": str(uuid.uuid4())})

    # Step 2: Create FAISS index with search results
    search_faiss_index = create_faiss_index(search_results1, search_results)
    pdf_faiss_index = create_pdf_faiss_index(pages)

    pdfdocRetriever = pdf_faiss_index.as_retriever()
    docRetriever = search_faiss_index.as_retriever()

    # Step 3: Retrieve context
    docs = docRetriever.invoke(user_query)
    pdfdocs = pdfdocRetriever.invoke(user_query)
    web_retrieved_context = "\n\n".join([doc.page_content for doc in docs])
    pdf_retrieved_content = "\n\n".join([doc.page_content for doc in pdfdocs])

    messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"Web Context:\n{web_retrieved_context}\n\nPdf Context:\n{pdf_retrieved_content}\n\nQuery: {user_query}\nAnswer:")
    ]
    response = llm(messages)

    documents.append(response.content)
    metadatas.append({"role": "system", "id": str(uuid.uuid4())})

    return response



# if __name__ == '__main__':
#     app.run(debug=True)

def user_info(name, level_choices, goal_choices, institution, styles):
    levels = [
        {"label": "Beginner", "description": "New to ASL, learning basic hand shapes and simple words."},
        {"label": "Intermediate", "description": "Has some experience, working on more complex signs and sentences."},
        {"label": "Advanced", "description": "Proficient in ASL, focusing on fluency and expression."},
        {"label": "Instructor", "description": "Teaches ASL to others, knowledgeable in techniques and pedagogy."}
    ]

    goals = [
        {"label": "Improve Vocabulary", "description": "Focused on learning new words and expanding vocabulary."},
        {"label": "Practice Sentences", "description": "Aiming to practice forming and signing complete sentences."},
        {"label": "Enhance Fluency", "description": "Aspires to achieve a natural and fluent signing style."},
        {"label": "Teach ASL", "description": "Interested in instructing and mentoring other ASL learners."}
    ]

    institutions = [
        {"label": "ASL Course", "description": "Enrolled in a structured ASL course."},
        {"label": "Community Center", "description": "Participating in ASL programs at a local community facility."},
        {"label": "Online ASL Course", "description": "Engaged in ASL training through virtual platforms."},
        {"label": "Self-Taught", "description": "Learning ASL through self-guided methods, such as online tutorials."}
    ]

    styles = [
        {"label": "Everyday Communication", "description": "Focusing on practical ASL for daily use."},
        {"label": "Storytelling", "description": "Practicing ASL for storytelling and expressive signing."},
        {"label": "Formal Signing", "description": "Learning ASL for formal settings, such as presentations and speeches."}
    ]

    formatted_string = f"""{name} is at the {levels[level_choices[0]]['label']} level, aiming to {goals[goal_choices[0]]['label']}, currently learning at {institutions[institution[0]]['label']}, and focusing on {styles[styles[0]]['label']} style."""
    return formatted_string

# Main function to answer query using web search and ChatQA with system instructions
def answer_query_with_rag(user_query, system_instructions):
    # Step 1: Get web search results
    search_results = google_search1(user_query)
    search_results1 = google_search("with regards to american sign language learning"+user_query)

    documents.append(user_query)
    metadatas.append({"role": "user", "id": str(uuid.uuid4())})

    # Step 2: Create FAISS index with search results
    faiss_index = create_faiss_index(search_results1, search_results)
    docRetriever = faiss_index.as_retriever()

    # Step 3: Retrieve context
    docs = docRetriever.invoke(user_query)
    retrieved_context = "\n\n".join([doc.page_content for doc in docs])
    print("retrieved context:"+retrieved_context)
    messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"Context:\n{retrieved_context}\n\nQuery: {user_query}\nAnswer:")
    ]
    response = llm(messages)

    documents.append(response.content)
    metadatas.append({"role": "system", "id": str(uuid.uuid4())})

    return response

# Example usage
# system_instructions = f"Please provide a detailed and comprehensive answer with actionable advice for improving ASL skills based on this user's profile: Include tips for improving letter, word, and sentence hand gesture formation, and emphasize the importance of facial expressions and body language."
# system_instructions1 = f"you are an ASL assistant that assists in users learning plans with bulleted out objectives for learning letters,words,and sentences"
# response = answer_query_with_rag("generate me an actionable american sign langauge learning plan", system_instructions1)
# print(response)
