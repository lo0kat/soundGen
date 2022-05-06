from model_creator.auto_encoder import Autoencoder
import numpy as np
from tensorflow.python.keras.backend import set_session
import tensorflow.compat.v1 as tf
import tensorflow

class Model:
  """
  This will allow you to upload the model and work with it
  """

  def __init__(self, folder_saved : str):
    self.sess = tf.Session()
    self.graph = tf.get_default_graph()
    set_session(self.sess)

    self.auto_encoder = Autoencoder.load(folder_saved)
    self.encoder = self.auto_encoder.encoder
    self.decoder = self.auto_encoder.decoder
  
  def encoder_predict(self, spectrogram: np.array, oiseau : str):

    with self.graph.as_default():
      set_session(self.sess)
      pred = self.encoder.predict(spectrogram)
    
    ref_mean = np.load("model_result/" + oiseau + ".npy")[0]

    dist = []
    for enregistrement in pred:
      dist.append(np.linalg.norm(ref_mean-enregistrement))
    
    return np.mean(dist)

  def decoder_predict(self, latent_vec: np.array):
    return self.decoder.predict(latent_vec)