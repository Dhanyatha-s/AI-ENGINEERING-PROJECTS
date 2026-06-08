"""
## BASIC RAG APPLICATION
------ Strcuture of RAG Building --------
|__ Install requirements.txt
|__ Import Libaries
|__ Prepare text dataset
|__ Load Models 
|__ Tokenize and Chunk
|__ Embed
|__ store in VectorDB
|__ Chain 
|__ User Query
|__ Retrieve
"""

# Import libaries
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ADD Text DATA

text = """OmniCorp Remote Work Policy 2026

1. Eligibility and Core Hours
All full-time employees who have completed their 90-day probationary period are eligible for the hybrid work model. Employees must be online and available during the company's core collaboration hours, which are defined as 10:00 AM to 3:00 PM EST, Tuesday through Thursday.

2. Equipment and Stipends
OmniCorp provides a one-time home office setup stipend of $500. This stipend can be used to purchase ergonomic chairs, monitors, or keyboards. Receipts must be submitted to the finance department via the ExpenseBot portal within 30 days of purchase. Monthly internet subsidies are capped at $60 per month.

3. Security Protocol
Employees accessing the internal database (Project Nexus) must connect through the SecureShield VPN at all times. Multi-factor authentication (MFA) via the AuthenticateMe app is mandatory for every login. Sharing company hardware with non-employees, including family members, is strictly prohibited and results in immediate disciplinary review.
"""
# Lets Convert text to langchain document
docs = [Document(page_content=text, metadata = {"source": "Experimental_text"})]

print("Successfully converted raw text to document")

# Chunking
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size = 150, 
    chunk_overlap=20
)
splits = text_spliter.split_documents(docs)
print(f"Successfully Splitted document into {len(splits)} no. of chunks ")

# Embedding and In_memory vector store
embeddings = OllamaEmbeddings(model = "nomic-embed-text")
vectorDB = FAISS.from_documents(splits, embeddings)

# Convert vector store into standardized retrieval object
retriever = vectorDB.as_retriever(search_kwargs = {'k':2})
print("Successfully retrieved at top-2 semantic search")

# Template
template = """You are a Precise Research assistant. Answers the Question based only on the context given bellow:
<context>
{content}
</context>

Question: {question}
Answer :

"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOllama(model = "llama3.1", temperature=0)

print("Models are SetUp")

# Assemble and Execute the RAG Chain
def format_doc(retriever):
    return "\n\n".join(doc.page_content for doc in retriever)

rag_chain = (
    {"content":retriever | format_doc, "question":RunnablePassthrough()}
    | prompt
    |llm
    |StrOutputParser()
)
# Run
query = input("Enter your question here")

response = rag_chain.invoke(query)

print("Running Your Query.....")
print(f"\n Bot response: {response} ")








