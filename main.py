from fastapi import FastAPI, Request
from pydantic import BaseModel
from detector import analyze_text

app = FastAPI(title="VeriText AI Essay Detector")

class TextPayload(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "online", "message": "VeriText AI Detector API is running!"}

@app.post("/analyze")
def analyze(payload: TextPayload):
    # Pass the user's text into our NLP detection engine in detector.py
    results = analyze_text(payload.text)
    return results