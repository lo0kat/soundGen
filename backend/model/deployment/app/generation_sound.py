from model_creator.auto_encoder import Autoencoder
from tensorflow.python.keras.backend import set_session
import tensorflow.compat.v1 as tf
from model_creator.preprocess import MinMaxNormaliser
from model_creator.config_default import HOP_LENGTH
import numpy as np
import librosa


class Generation_Oiseau :
    """
    Permet de tirer un vecteur latent et générer un son à partir
    """

    def __init__(self, path_saved = './model'):
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()
        set_session(self.sess)

        trained_auto_encoder = Autoencoder.load(path_saved)
        self.trained_decoder = trained_auto_encoder.decoder
        
    
    def __tirage_latent_space(self) -> np.array:
        latent_diff = []
        for mu, var in zip(self.vecteur_moyen[0], self.vecteur_moyen[1]):
            latent_diff.append(np.random.normal(mu, np.sqrt(var), 1))

        return np.array(latent_diff).reshape(1,256)

    def creation_spectrogram(self) -> None:
        latent_space = self.__tirage_latent_space()
        with self.graph.as_default():
            set_session(self.sess)
            pred = self.trained_decoder.predict(latent_space)
        return pred

    def convert_spectrograms_to_audio(self, min_max_value):

        log_spectrogram = self.spectrogram.reshape(1024,112)
        #apply denormalisation
        denorm_log_spec = self._min_max_normaliser.denormalise(log_spectrogram, min_max_value["min"], min_max_value["max"])
        #log spectrogram
        spec = librosa.db_to_amplitude(denorm_log_spec)
        #apply Griffin-Lin
        signal = librosa.istft(spec, hop_length=HOP_LENGTH)
        return signal

    def gen_sound(self, oiseau:str) -> dict:
        self.vecteur_moyen = np.load('model_result/' + oiseau + '.npy')
        self._min_max_normaliser = MinMaxNormaliser(0, 1)
        self.spectrogram = self.creation_spectrogram()
        return {oiseau: str(list(self.convert_spectrograms_to_audio({"min":-30,"max":30})))}