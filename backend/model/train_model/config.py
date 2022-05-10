LIEN_METADATA = "metadata.csv"
LIEN_DIR_MP3 = "mp3/"

# 10 oiseaux -> 12.5H 
NB_ESPECE = 10


#Paramètre modèle
BATCH_SIZE = 16
EPOCH = 30
TRIAL = 25


#Parametre pour la découpe
#Temps du silence
SILENCE_GAP = 500
#Niveau de son du silence
SILENCE_BAR = -32
#Too short lenght for sing
TOO_SHORT_LENGHT = 1.1
#Too long sing
TOO_LONG_LENGHT = 5

tuning_dico = {
    'classique':{
        "conv_filters":(512,256, 128, 64, 32),
        "conv_kernels":(3,3,3,3,2),
        "conv_strides":(2,2,2,2, (2,1)),
    },
    'reduit_taille':{
        "conv_filters":(512,256, 128, 64),
        "conv_kernels":(3,3,3,3),
        "conv_strides":(2,2,2, (2,1)),
    },
    'change_kernel':{
        "conv_filters":(512,256, 128, 64, 32),
        "conv_kernels":(2,2,2,2,2),
        "conv_strides":(2,2,2,2, (2,1)),
    },
    'classique_max':{
        "conv_filters":(1024,512, 256, 128, 64),
        "conv_kernels":(3,3,3,3,2),
        "conv_strides":(2,2,2,2, (2,1)),
    },
    'reduit_strenght':{
        "conv_filters":(256,128, 64, 32, 8),
        "conv_kernels":(3,3,3,3,2),
        "conv_strides":(2,2,2,2, (2,1)),
    },
}