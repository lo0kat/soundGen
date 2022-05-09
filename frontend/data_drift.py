import pandas as pd

class Data_Drift:
    """
    Creation d'une classe permettant de gérer les changements de données
    """

    def __init__(self, log_flag:str):
        self.df_log = pd.read_csv(log_flag)

    def retrain_on_number_flags(self, limite:int):
        if self.df_log.shape[0]>limite:
            return True
        else:
            False



if __name__ == "__main__":
    Data_Drift("flagged/log.csv")