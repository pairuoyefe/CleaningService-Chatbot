# ============================================================
# api.py  —  Reis Cleaning Services · FastAPI App
# ============================================================
# Run with:  uvicorn api:app --reload
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from reis_core import build_vectorstore, build_llm, query_llm, place_order

# ------------------------------
# Startup: load models once
# ------------------------------

resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Loading vectorstore and LLM...")
    resources["vectorstore"] = build_vectorstore()
    resources["llm"]         = build_llm()
    print("✅ Ready!")
    yield
    resources.clear()

app = FastAPI(
    title="Reis Cleaning Services API",
    description="AI-powered customer query and order placement API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allows React (port 5173) to talk to FastAPI (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Request / Response Schemas
# ------------------------------

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    wants_order: bool

class OrderRequest(BaseModel):
    name: str
    phone_number: str
    service_requested: str
    notes: str = ""

class OrderResponse(BaseModel):
    success: bool
    message: str
    ticket: dict

# ------------------------------
# Endpoints
# ------------------------------

@app.get("/")
def root():
    return {"message": "Reis Cleaning Services API is running 🧹"}


@app.post("/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    vectorstore = resources.get("vectorstore")
    llm         = resources.get("llm")

    if not vectorstore or not llm:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    answer = query_llm(req.query, vectorstore, llm)
    wants_order = "would you like to place an order" in answer.lower()

    return QueryResponse(answer=answer, wants_order=wants_order)


@app.post("/order", response_model=OrderResponse)
def place_order_endpoint(req: OrderRequest):
    if not req.name.strip() or not req.phone_number.strip() or not req.service_requested.strip():
        raise HTTPException(status_code=400, detail="name, phone_number, and service_requested are required.")

    ticket = place_order(
        name    = req.name,
        phone   = req.phone_number,
        service = req.service_requested,
        notes   = req.notes
    )

    return OrderResponse(
        success = True,
        message = f"Order placed for {req.name}. Email sent to agent.",
        ticket  = ticket
    )
