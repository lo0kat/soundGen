from ray import serve
import numpy as np

@serve.deployment(route_prefix="/auto_encoder")
class Auto_Encoder_Model:
    def __init__(self, model_path):
        import tensorflow as tf

        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)

    async def preprocess(self, starlette_request):
        return np.array((await starlette_request.json())["song"])

    async def __call__(self, starlette_request):
        # Step 1: transform HTTP request -> tensorflow input
        # Here we define the request schema to be a json array.
        input_array_preprocessed = self.preprocess(starlette_request)

        # Step 2: tensorflow input -> tensorflow output
        prediction = self.model(input_array_preprocessed)

        # Step 3: tensorflow output -> web output
        return {"prediction": prediction.numpy().tolist(), "file": self.model_path}

serve.start()
Auto_Encoder_Model.deploy("../model/weights.h5")