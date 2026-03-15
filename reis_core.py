# ============================================================
# reis_core.py  —  Reis Cleaning Services · Core Logic
# ============================================================

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ------------------------------
# Configuration
# ------------------------------

os.environ["GROQ_API_KEY"] = ""

SMTP_SENDER_EMAIL    = "zimorworld@gmail.com"
SMTP_SENDER_PASSWORD = ""
SMTP_AGENT_EMAIL     = "pairuoyefe@gmail.com"
SMTP_HOST            = "smtp.gmail.com"
SMTP_PORT            = 465
TICKET_FILE          = "support_ticket.json"
PDF_PATH             = "Reis_Cleaning_Services_Guide.pdf"

# ------------------------------
# Prompt
# ------------------------------

cleaning_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a professional cleaning services assistant.
Rules:
1. Answer the customer's question using ONLY the provided context (cleaning services documents).
2. Respond clearly and professionally with detailed explanations.
3. After answering, ask: "Would you like to place an order for our services?"
4. If customer says 'yes':
   - Immediately ask for their name and phone number.
   - When received, return ONLY valid JSON:
     {{"name": "", "phone_number": "", "service_requested": "", "notes": ""}}
   - Use the original question as "service_requested" and notes for any additional comments.
5. Do NOT include explanations when returning JSON.
"""),
    ("human", "Context:\n{context}\n\nCustomer Question:\n{input}")
])

# ------------------------------
# Setup: Load PDF & Build Vectorstore
# ------------------------------

def build_vectorstore(pdf_path: str = PDF_PATH):
    """Load PDF, chunk it, embed and return FAISS vectorstore."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"trust_remote_code": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Vector store created")
    return vectorstore


def build_llm():
    """Initialise and return the Groq LLM."""
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

# ------------------------------
# Email / Ticket
# ------------------------------

def send_ticket_email(ticket_data: dict):
    """Send ticket JSON as email to agent using SSL."""
    try:
        msg = MIMEMultipart()
        msg["From"]    = SMTP_SENDER_EMAIL
        msg["To"]      = SMTP_AGENT_EMAIL
        msg["Subject"] = f"New Cleaning Service Ticket: {ticket_data['service_requested']}"
        msg.attach(MIMEText(json.dumps(ticket_data, indent=4), "plain"))

        # Using SSL on port 465 instead of TLS on 587
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SMTP_SENDER_EMAIL, SMTP_SENDER_PASSWORD)
            server.sendmail(SMTP_SENDER_EMAIL, SMTP_AGENT_EMAIL, msg.as_string())
        print("✅ Ticket sent via email.")

    except smtplib.SMTPAuthenticationError:
        print("❌ EMAIL ERROR: Authentication failed. Check your Gmail app password.")
        raise
    except smtplib.SMTPException as e:
        print(f"❌ EMAIL ERROR: SMTP error — {e}")
        raise
    except Exception as e:
        print(f"❌ EMAIL ERROR: Unexpected error — {e}")
        raise


def save_ticket(ticket_data: dict):
    """Save ticket to local JSON file."""
    with open(TICKET_FILE, "w", encoding="utf-8") as f:
        json.dump(ticket_data, f, ensure_ascii=False, indent=4)

# ------------------------------
# Core Query Handler
# ------------------------------

def query_llm(query: str, vectorstore, llm) -> str:
    """Retrieve context and invoke LLM. Returns raw answer string."""
    docs = vectorstore.similarity_search_with_score(query, k=3)
    context = "\n\n".join([doc.page_content for doc, _ in docs]) if docs else ""
    chain = cleaning_prompt | llm
    response = chain.invoke({"context": context, "input": query})
    return response.content


def place_order(name: str, phone: str, service: str, notes: str = "") -> dict:
    """Build ticket, save it, and email it. Returns the ticket dict."""
    ticket = {
        "name": name,
        "phone_number": phone,
        "service_requested": service,
        "notes": notes
    }
    save_ticket(ticket)
    send_ticket_email(ticket)
    return ticket