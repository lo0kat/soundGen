from model_creator.auto_encoder import Autoencoder
import ray
from ray import serve

@serve.deployment
class Encoder:
  def __init__(self):
    self.auto_encoder = Autoencoder.load("model_trained")
    self.encoder = self.auto_encoder.encoder
    self.decoder = self.auto_encoder.decoder

  def __call__(self, request):
    
    return {"forecast": self.encoder.predict(request)}

if __name__ == "__main__":
  ray.init()
  serve.start()
  Encoder.deploy()