from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from detector import analyze_text

app = FastAPI(title="VeriText AI Essay Detector")


templates = Jinja2Templates(directory="templates")


class TextPayload(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
   
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
def analyze(payload: TextPayload):
    # Process text using our statistical engine in detector.py
    results = analyze_text(payload.text)
    return results