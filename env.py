import pandas as pd
import numpy as np

class MarketingEnv:
    def __init__(self, csv_path="dataset_preparado.csv"):
        
        self.df = pd.read_csv(csv_path)

        # colunas do estado
        self.state_cols = [
            "idade",
            "genero",
            "renda",
            "visitas_site",
            "paginas_por_visita",
            "tempo_no_site",
            "compras_anteriores",
            "pontos_fidelidade"
        ]

        # acoes disponiveis
        self.actions = {
            0: "nenhuma_acao",
            1: "email",
            2: "anuncio",
            3: "rede_social"
        }

        #custos das acoes
        self.action_cost = {
            0: 0.00,
            1: 0.10,
            2: 0.25,
            3: 0.20
        }

        
        self.current_index = None

    def reset(self):
        self.current_index = np.random.randint(0, len(self.df))
        return self._get_state()

    def _get_state(self):
        return self.df.loc[self.current_index, self.state_cols].values

    def step(self, action):
        cliente = self.df.loc[self.current_index]
        conversao = cliente["conversao"]   
        custo = self.action_cost[action]
        reward = conversao - custo
        self.current_index = np.random.randint(0, len(self.df))
        next_state = self._get_state()

        done = False   
        info = {}

        return next_state, reward, done, info
