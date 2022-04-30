LIEN_METADATA = "data/metadata.csv"
LIEN_DIR_MP3 = "data/mp3/"
NB_ESPECE = 1

#Parametre pour la découpe

#Temps du silence
SILENCE_GAP = 500
#Niveau de son du silence
SILENCE_BAR = -32
#Too short lenght for sing
TOO_SHORT_LENGHT = 1.1
#Too long sing
TOO_LONG_LENGHT = 5

model_param = {
    'learning_rate' : 0.00025,
    'batch_size' : 12,
    'epochs' : 20,
}