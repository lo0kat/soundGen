from encoder import Model
import uvicorn
from fastapi import FastAPI
import numpy as np
import tensorflow.compat.v1 as tf


model = Model("model_trained")
app = FastAPI()

def mise_for_prediction(input_user: list):
  input_user = np.array(input_user)
  input_user = np.expand_dims(input_user, 3)

  return input_user

#Main road to know the main road of the API
@app.get("/")
def read_root():
  return {"message": "Welcome from the API model"}

@app.post("/forecast_encoder")
def get_pred_encoder(input_user : dict) -> dict:

  res_dico = {}
  for joueur in input_user:
    res_dico[joueur] = str(model.encoder_predict(mise_for_prediction(input_user[joueur]), "Fringilla_coelebs"))
  
  return res_dico

@app.post("/forecast_decoder")
def get_pred_decoder(input_user : dict) -> dict:
  return model.decoder_predict(input_user)

if __name__ == "__main__":
  uvicorn.run("my_deployement:app", host="0.0.0.0", port=8082, reload = True)