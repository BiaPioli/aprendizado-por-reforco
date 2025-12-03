import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("digital_marketing_campaign_dataset.csv")

#renomear colunas 
df = df.rename(columns={
    "CustomerID": "cliente_id",
    "Age": "idade",
    "Gender": "genero",
    "Income": "renda",
    "CampaignChannel": "canal_campanha",
    "CampaignType": "tipo_campanha",
    "AdSpend": "gasto_anuncio",
    "ClickThroughRate": "ctr",
    "ConversionRate": "taxa_conversao",
    "WebsiteVisits": "visitas_site",
    "PagesPerVisit": "paginas_por_visita",
    "TimeOnSite": "tempo_no_site",
    "SocialShares": "compartilhamentos_sociais",
    "EmailOpens": "aberturas_email",
    "EmailClicks": "cliques_email",
    "PreviousPurchases": "compras_anteriores",
    "LoyaltyPoints": "pontos_fidelidade",
    "AdvertisingPlatform": "plataforma_anuncio",
    "AdvertisingTool": "ferramenta_anuncio",
    "Conversion": "conversao"
})

print("colunas traduzidas:")
print(df.columns)

#selecionar colunas do estado 
state_cols = [
    "idade",
    "genero",
    "renda",
    "visitas_site",
    "paginas_por_visita",
    "tempo_no_site",
    "compras_anteriores",
    "pontos_fidelidade"
]

#selecionar coluna alvo 
target_col = "conversao"

#criando dataset novo
data_novo = df[state_cols + [target_col]].copy()
data_novo.to_csv("dataset_preparado.csv", index=False)


#converter variáveis categóricas
categorical_cols = ["genero"]

encoder = LabelEncoder()

for col in categorical_cols:
    data_novo[col] = encoder.fit_transform(data_novo[col])

print("primeiras linhas do dataset preparado:")
print(data_novo.head())

print("tipos de dados após preparo:")
print(data_novo.dtypes)

