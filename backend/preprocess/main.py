import uvicorn
from fastapi import FastAPI
import preprocess

app = FastAPI()

#Main road to know the main road of the API
@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/preprocess")
def get_pred(input_user : dict) -> dict:
    return preprocess.preprocess_file(input_user)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)