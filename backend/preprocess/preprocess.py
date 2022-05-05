<<<<<<< HEAD:backend_preprocess/preprocess.py
from preprocess_class import Preprocess_Aplicatif
=======
from pydub import AudioSegment
from pydub.silence import split_on_silence
import ast
import soundfile as sf
import numpy as np
>>>>>>> main:backend/preprocess/preprocess.py

def preprocess_file(user_data: dict) -> dict:
    """
    Permet de renvoyer un json avec les différents segments audios décomposés
    """

<<<<<<< HEAD:backend_preprocess/preprocess.py
    json_output_preprocess = {}
    for key in user_data:
        preprocess_applicatif = Preprocess_Aplicatif(user_data[key])
        json_output_preprocess[key] = preprocess_applicatif.find_spectrogram()
=======
    sr_original = int(input_user[0])
    song = np.array(ast.literal_eval(input_user[1]), dtype='int32')
    sf.write('records/record.wav', song, sr_original)

    # Load your audio.
    song = AudioSegment.from_wav('records/record.wav')

    # Split track where the silence is 2 seconds or more and get chunks using 
    # the imported function.
    chunks = split_on_silence (
        # Use the loaded audio.
        song, 
        # Specify that a silent chunk must be at least 500 ms
        min_silence_len = 500,
        # Consider a chunk silent if it's quieter than -16 dBFS.
        # (You may want to adjust this parameter.)
        silence_thresh = -32
    )
    
    return chunks

def reconstruct_chunks(chunks, padding = 300):
    """
    Permet de reconstruire les différents chunks 
    """
    
    # Define a function to normalize a chunk to a target amplitude.
    def match_target_amplitude(aChunk, target_dBFS):
        ''' Normalize given audio chunk '''
        change_in_dBFS = target_dBFS - aChunk.dBFS
        return aChunk.apply_gain(change_in_dBFS)

    # Process each chunk with your parameters
    dico_res = {}
    for i, chunk in enumerate(chunks):
        # Create a silence chunk that's 0.5 seconds (or 1000 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=padding)

        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Normalize the entire chunk.
        normalized_chunk = match_target_amplitude(audio_chunk, -20.0)
        # Export the audio chunk with new bitrate.
        dico_res[str(i)] = str(list(normalized_chunk.get_array_of_samples()))
        '''
        normalized_chunk.export(
            "records/{0}.wav".format(i),
            bitrate = "192k",
            format = "wav"
        )
        '''
    return dico_res

def preprocess_file(user_data: dict) -> dict:
    """
    Permet de renvoyer un json avec les différents segments audios décomposés
    """

    json_output_preprocess = {}
    for key in user_data:
        chunks = find_chunks(user_data[key])
        json_output_preprocess[key] = reconstruct_chunks(chunks)
>>>>>>> main:backend/preprocess/preprocess.py
    return json_output_preprocess