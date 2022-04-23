import gradio as gr
import requests
import ast

def greet(text1:str, audio1:tuple, text2:str, audio2:tuple):
    #Transform the audio
    json_original = {
        text1: [str(audio1[0]), str(list(audio1[1]))],
        text2: [str(audio2[0]), str(list(audio2[1]))]
        }
    #Send samples to the model
    res = requests.post("http://127.0.0.1:8000/preprocess", json=json_original)
    dico_res = ast.literal_eval(res.text)

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