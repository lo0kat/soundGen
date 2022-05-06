import ray
from ray import serve
from fastapi import FastAPI
from model_creator.auto_encoder import Autoencoder

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class Counter:
  def __init__(self):
    self.auto_encoder = Autoencoder.load("model_trained")
    self.encoder = self.auto_encoder.encoder
    self.decoder = self.auto_encoder.decoder

  @app.get("/")
  def get(self):
    return {"Hello World from model !!"}

  @app.get("/encoder_pred")
  def encod_forecast(self, request):
    return {"forecast": self.encoder.predict(request)}

if __name__ == "__main__":
    ray.init()
    serve.start(detached=True)
    Counter.deploy()