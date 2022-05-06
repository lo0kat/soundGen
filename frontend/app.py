import gradio as gr
import requests
import ast


def reconvertion_request_res(res):
    """
    Change the request text into number dictionnary
    """
    
    first_trad = ast.literal_eval(res.text)

    return first_trad


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
    res_preprocess = requests.post("http://0.0.0.0:8080/preprocess", json=json_original)

    #Send for prediction
    res_pred = requests.post("http://0.0.0.0:8082/forecast_encoder", json=reconvertion_request_res(res_preprocess))

    #Convert the response into numbers
    #res_converted = reconvertion_request_res(res_pred)

    return "Hello World"


#Permet l'enregistrement du son et sa modification de façon interactive
text1 = gr.inputs.Textbox(type="str", label="Player 1 Name")
audio1 = gr.inputs.Audio(source="microphone", label='Enregistrement 1', optional=False)
text2 = gr.inputs.Textbox(type="str", label="Player 2 Name")
audio2 = gr.inputs.Audio(source="microphone", label='Enregistrement 2', optional=False)
iface = gr.Interface(fn=greet,
                    inputs=[text1, audio1, text2, audio2], 
                    outputs="text",
                    title="Who is the fake bird",
                    description="This is made to see who will imitate the best the birds of hese choice.",)


if __name__ == "__main__":
    app, local_url, share_url = iface.launch()