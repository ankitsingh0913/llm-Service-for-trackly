from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from rag_module import get_insight
from report_module import generate_pdf_report

app = FastAPI()

class InsightQuery(BaseModel):
    transactions: List[Dict]
    question: str

class ReportQuery(BaseModel):
    transactions: List[Dict]

@app.post("/rag")
def ask_ai(data: InsightQuery):
    answer = get_insight(data.transactions, data.question)
    return {"answer": answer}

@app.post("/generate-report")
def generate_report(data: ReportQuery):
    path = generate_pdf_report(data.transactions)
    return {"report_path": path}
