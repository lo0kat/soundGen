from preprocess_class import Preprocess_Aplicatif

def preprocess_file(user_data: dict) -> dict:
    """
    Permet de renvoyer un json avec les différents segments audios décomposés
    """

    json_output_preprocess = {}
    for key in user_data:
        preprocess_applicatif = Preprocess_Aplicatif(user_data[key], key)
        json_output_preprocess[key] = preprocess_applicatif.find_spectrogram().tolist()
    return json_output_preprocess