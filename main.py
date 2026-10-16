from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello JobShield"}

@app.get("/health")
def health():
    return {"ok": True}