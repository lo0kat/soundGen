import gradio as gr
import requests
import ast
import pandas as pd
from correspondence_oiseau import DICO_CORR

def greet(oiseau:str, text1:str, audio1:tuple, text2:str, audio2:tuple) -> tuple:
    """
    Allow you to do :
        1) The preprocessing
        2) Compute the spectrogram
        3) Make the prediction
    """

    #Prepare data into json for sending
    json_original = {
        text1: [str(audio1[0]), str(list(audio1[1]))],
        text2: [str(audio2[0]), str(list(audio2[1]))]
        }

    #Send request to the FastAPI for preprocessing
    res_preprocess = requests.post("http://0.0.0.0:8080/preprocess", json=json_original)

    #Ajout du type d'oiseau pour la prédiction
    res_preprocess_tranform = ast.literal_eval(res_preprocess.text)
    res_preprocess_tranform["Oiseau"] = DICO_CORR[oiseau]

    #Send for prediction
    res_pred = requests.post("http://0.0.0.0:8082/forecast_encoder", json=res_preprocess_tranform)

    return pd.DataFrame(ast.literal_eval(res_pred.text), index = [1])


#Permet l'enregistrement du son et sa modification de façon interactive
text1 = gr.inputs.Textbox(type="str", 
                        label="Player 1 Name")

audio1 = gr.inputs.Audio(source="microphone", 
                        label='First recording', 
                        optional=False)

text2 = gr.inputs.Textbox(type="str", 
                        label="Player 2 Name")


audio2 = gr.inputs.Audio(source="microphone", 
                        label='Second recording', 
                        optional=False)

choix_oiseau = gr.inputs.Radio(["Pinson des arbres", "Canard", "Autruche", "Coucou"],
                                label='Wich bird bird would you like to imitate ?')

iface = gr.Interface(fn=greet,
                    inputs=[choix_oiseau, text1, audio1, text2, audio2], 
                    outputs="dataframe",
                    title="Who is the fake bird",
                    description="This is made to see who will imitate the best the birds of hese choice.",
                    theme="peach",
                    )


if __name__ == "__main__":
    app, local_url, share_url = iface.launch()