from model_creator.auto_encoder import Autoencoder
import numpy as np
from tensorflow.python.keras.backend import set_session
import tensorflow.compat.v1 as tf

class Model:
  """
  This will allow you to upload the model and work with it
  """

  def __init__(self, folder_saved : str):

    self.sess = tf.Session()
    #This is a global session and graph
    self.graph = tf.get_default_graph()
    set_session(self.sess)

    self.auto_encoder = Autoencoder.load(folder_saved)
    self.encoder = self.auto_encoder.encoder
    self.decoder = self.auto_encoder.decoder
  
  def encoder_predict(self, spectrogram: np.array):

    with self.graph.as_default():
      set_session(self.sess)
      pred = self.encoder.predict(spectrogram)
    
    return pred

  def decoder_predict(self, latent_vec: np.array):
    return self.decoder.predict(latent_vec)