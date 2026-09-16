import pandas as pd
from scipy.stats import norm

# Carregar a base
df = pd.read_csv("world_oecd_revenue_statistics_revenue.csv")

# Converter para número; valores que não forem números viram NaN
dados = pd.to_numeric(df["value_usd"], errors="coerce")

# Manter somente números maiores que 0
dados = dados[dados > 0]

# Mediana
mediana = dados.median()

# Média e desvio-padrão
media = dados.mean()
desvio = dados.std()

# Probabilidade acima da mediana
probabilidade = 1 - norm.cdf(
    mediana,
    loc=media,
    scale=desvio
)

# Classificação
if probabilidade < 0.25:
    classificacao = "Raro"
elif probabilidade < 0.50:
    classificacao = "Pouco provável"
elif probabilidade < 0.75:
    classificacao = "Provável"
else:
    classificacao = "Quase certo"

# Resultados
print("ANALISE MEDIANA\n")
print(f"Quantidade de valores válidos: {len(dados)}")
print(f"Mediana: US$ {mediana:,.2f}")
print(f"Probabilidade acima da mediana: {probabilidade:.2%}")
print(f"Classificação: {classificacao}\n")

media = dados.mean()
desvio = dados.std()

limite_inferior = media - 2 * desvio
limite_superior = media + 2 * desvio

probabilidade = norm.cdf(limite_superior, media, desvio) - \
                norm.cdf(limite_inferior, media, desvio)

if probabilidade < 0.25:
    classificacao = "Raro"
elif probabilidade < 0.50:
    classificacao = "Pouco provável"
elif probabilidade < 0.75:
    classificacao = "Provável"
else:
    classificacao = "Quase certo"

print(f"ANALISE MEDIA\n")
print(f"Média: {media:.2f}")
print(f"Desvio-padrão: {desvio:.2f}")
print(f"Intervalo: [{limite_inferior:.2f}, {limite_superior:.2f}]")
print(f"Probabilidade de se obter numero maior : {probabilidade:.2%}")
print(f"Classificação: {classificacao}\n")

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("world_oecd_revenue_statistics_revenue.csv")

# Transformar value_usd em número
df["value_usd"] = pd.to_numeric(df["value_usd"], errors="coerce")

# Remover NaN e manter somente value_usd > 0
df = df.dropna(subset=["value_usd"])
df = df[df["value_usd"] > 0]

# Remover NaN da variável X também
df = df.dropna(subset=["year"])

# Separar X e y
X = df[["year"]]
y = df["value_usd"]

# Criar e treinar o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Previsões
y_pred = modelo.predict(X)

print("Coeficiente:", modelo.coef_[0])
print("Intercepto:", modelo.intercept_)
print("R²:", modelo.score(X, y))

# Gráfico
plt.scatter(X, y, label="Dados")
plt.plot(X, y_pred, label="Reta ajustada")

plt.xlabel("year")
plt.ylabel("value_usd")
plt.title("Regressão Linear")
plt.legend()
plt.show()