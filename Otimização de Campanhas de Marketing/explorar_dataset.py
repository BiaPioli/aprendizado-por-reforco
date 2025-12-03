import pandas as pd 

df = pd.read_csv("digital_marketing_campaign_dataset.csv")

print("primeiras linhas do dataset:")
print(df.head())

print("colunas")
print(df.columns)

print("info:")
print(df.info())

# Estatísticas para colunas numéricas
print("estatistica:")
print(df.describe())

# Ver valores únicos nas colunas categóricas
print("valores unicos por colunas:")
for col in df.columns:
    print(f"\nColuna: {col}")
    print(df[col].unique())
