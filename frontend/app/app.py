import gradio as gr
import requests
import ast
import pandas as pd
import numpy as np
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
    #res_preprocess = requests.post("http://soundgen_preprocessing-api_1:8080/preprocess", json=json_original)

    #Ajout du type d'oiseau pour la prédiction
    res_preprocess_tranform = ast.literal_eval(res_preprocess.text)
    res_preprocess_tranform["Oiseau"] = DICO_CORR[oiseau]

    #Send for prediction
    res_pred = requests.post("http://0.0.0.0:8082/forecast_encoder", json=res_preprocess_tranform)
    # res_pred = requests.post("http://soundgen_prediction-api_1:8082/forecast_encoder", json=res_preprocess_tranform)

    #Generate sound
    gen_res = []
    for _ in range(2):
        res_gene = requests.post("http://0.0.0.0:8082/forecast_decoder", json={"Oiseau":DICO_CORR[oiseau]})
        # res_gene = requests.post("http://soundgen_prediction-api_1:8082/forecast_decoder", json={"Oiseau":DICO_CORR[oiseau]})
        gen_res.append(np.array(ast.literal_eval(ast.literal_eval(res_gene.text)[DICO_CORR[oiseau]]), dtype='int16'))

    return pd.DataFrame(ast.literal_eval(res_pred.text), index = [1]), (22050, gen_res[0]), (22050, gen_res[1])

# Donne la liste des noms des espèces en Français
species_list = list(DICO_CORR.keys())


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

choix_oiseau = gr.inputs.Radio(species_list,
                                label='Wich bird bird would you like to imitate ?')

iface = gr.Interface(fn=greet,

                    inputs=[choix_oiseau, text1, audio1, text2, audio2], 
                    outputs=["dataframe", "audio", "audio"],

                    title="Who is the fake bird",
                    description='''This is our final project of my engineering school. It has been made in one month with an other student
                                who works in cloud computing. The idea was to generate birds songs with an Artificial Intelligence. We 
                                started with a tutorial from Valerio Velardo on sound VAE. We changed a bit the model to make it more 
                                more complex. We trained the model on Deep Learning machins because we needed GPU. Moreover, we tried 
                                different achitectures to find the best model.''',

                    theme="peach",
                    )


if __name__ == "__main__":
    print(species_list)
    app, local_url, share_url = iface.launch(server_name="0.0.0.0")