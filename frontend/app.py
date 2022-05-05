import gradio as gr
import requests
import ast
import numpy as np


def reconvertion_request_res(res):
    """
    Change the request text into number dictionnary
    """

    first_trad = ast.literal_eval(res.text)
    second_trad = {}
    for i in first_trad:
        for j in first_trad[i]:
            second_trad[i] = {int(j):ast.literal_eval(first_trad[i][j])}
    
    return second_trad


def greet(text1:str, audio1:tuple, text2:str, audio2:tuple) -> tuple:
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
    res = requests.post("http://127.0.0.1:8000/preprocess", json=json_original)
    #Convert the response into numbers
    res_converted = reconvertion_request_res(res)
    nb = list(res_converted[text1].keys())[0]
    song = np.array(res_converted[text1][nb], dtype='int16')
    sr = 48000

    #Send samples to the model

    return (sr,song)


#Permet l'enregistrement du son et sa modification de façon interactive
text1 = gr.inputs.Textbox(type="str", label="Player 1 Name")
audio1 = gr.inputs.Audio(source="microphone", label='Enregistrement 1', optional=False)
text2 = gr.inputs.Textbox(type="str", label="Player 2 Name")
audio2 = gr.inputs.Audio(source="microphone", label='Enregistrement 2', optional=False)
iface = gr.Interface(fn=greet,
                    inputs=[text1, audio1, text2, audio2], 
                    outputs="audio",
                    title="Who is the fake bird",
                    description="This is made to see who will imitate the best the birds of hese choice.",)


if __name__ == "__main__":
    app, local_url, share_url = iface.launch()