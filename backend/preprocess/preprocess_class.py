from pydub import AudioSegment
from pydub.silence import split_on_silence
import ast
import soundfile as sf
import os
import numpy as np
from model_creator.preprocess import Loader, Padder, LogSpectrogramExtractor, MinMaxNormaliser, PreprocessingPipeline
import model_creator.config_default as conf
import librosa

class Preprocess_Aplicatif:

    """
    Permet de découper l'audio reçu pour le preprocesser
    """

    def __init__(self, input_user: tuple, name : str):
        self.input_user = input_user
        self.name = name
        self.min_silence_len = 500
        self.silence_thresh = -32
        self.tmp_dir = "tmp_preprocess/"

        self.chunks = self.find_chunks()
        self.reconstruct_chunks()

    def nettoyage(self):
        """
        Permet de nettoyer le dossier temporaire où sont déposé les différents fichiers avant d'être déposé
        """

        for file in os.listdir(self.tmp_dir):
            os.remove(self.tmp_dir + file)

    def find_chunks(self):
        """
        Permet de trouver les différents chunks où les oiseaux chantes
        """

        sr_original = int(self.input_user[0])
        song = np.array(ast.literal_eval(self.input_user[1]), dtype='float32')
        song = librosa.resample(song, orig_sr=48000, target_sr=22050)
        
        sf.write(self.tmp_dir + 'record.wav', song, sr_original)

        # Load your audio.
        song = AudioSegment.from_wav(self.tmp_dir + 'record.wav')

        # Split track where the silence is 2 seconds or more and get chunks using 
        # the imported function.
        chunks = split_on_silence (
            # Use the loaded audio.
            song, 
            # Specify that a silent chunk must be at least 500 ms
            min_silence_len = self.min_silence_len,
            # Consider a chunk silent if it's quieter than -16 dBFS.
            # (You may want to adjust this parameter.)
            silence_thresh = self.silence_thresh
        )
        self.nettoyage()

        return chunks

    def find_spectrogram(self):
        """
        From array find the corresponding spectrogram
        """

        loader = Loader(conf.SAMPLE_RATE, conf.DURATION, conf.MONO)
        padder = Padder()
        log_spectrogram_extractor = LogSpectrogramExtractor(conf.FRAME_SIZE, conf.HOP_LENGTH)
        min_max_normaliser = MinMaxNormaliser(0, 1)

        preprocessing_pipeline = PreprocessingPipeline()
        preprocessing_pipeline.loader = loader
        preprocessing_pipeline.padder = padder
        preprocessing_pipeline.extractor = log_spectrogram_extractor
        preprocessing_pipeline.normaliser = min_max_normaliser

        spectrograms = np.array(preprocessing_pipeline.process_applicatif(self.tmp_dir))
        self.nettoyage()

        return spectrograms

    def reconstruct_chunks(self, padding = 300) -> None:

        """
        Permet de reconstruire les différents chunks 
        """
        
        # Define a function to normalize a chunk to a target amplitude.
        def match_target_amplitude(aChunk, target_dBFS):
            ''' Normalize given audio chunk '''
            change_in_dBFS = target_dBFS - aChunk.dBFS
            return aChunk.apply_gain(change_in_dBFS)

        # Process each chunk with your parameters
        for i, chunk in enumerate(self.chunks):
            # Create a silence chunk that's 0.5 seconds (or 1000 ms) long for padding.
            silence_chunk = AudioSegment.silent(duration=padding)

            # Add the padding chunk to beginning and end of the entire chunk.
            audio_chunk = silence_chunk + chunk + silence_chunk

            # Normalize the entire chunk.
            normalized_chunk = match_target_amplitude(audio_chunk, -20.0)
            
            normalized_chunk.export(
                ".//"+ self.tmp_dir + self.name +"{0}.wav".format(i),
                bitrate = "192k",
                format = "wav"
            )