# Importações (mantém as que já tem)
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import OneHotEncoder, StandardScaler  
from sklearn.pipeline import Pipeline  
from sklearn.compose import ColumnTransformer  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error  

# Criando um DataFrame de exemplo com preços de carros
data = {
    'marca': ['Toyota', 'Ford', 'BMW', 'Honda', 'Chevrolet'],
    'ano': [2015, 2018, 2020, 2017, 2016],
    'quilometragem': [50000, 30000, 10000, 40000, 60000],
    'preco': [45000, 60000, 120000, 55000, 50000]
}

df = pd.DataFrame(data)  

# Exibir os primeiros dados
print(df.head())  

# Separando as variáveis independentes e dependente
X = df[['marca', 'ano', 'quilometragem']]
y = df['preco']

# Transformando a variável categórica 'marca' em números com OneHotEncoder
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X[['marca']]).toarray()  # Converte em array numérico

# Criando um novo DataFrame combinando os dados transformados
X_transformed = np.hstack((X_encoded, X[['ano', 'quilometragem']].values))

# Exibir o novo conjunto de dados processado
print("Após transformação:\n", X_transformed)
# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Criando o modelo de regressão linear
modelo = LinearRegression()

# Treinando o modelo
modelo.fit(X_train, y_train)

# Fazendo previsões
y_pred = modelo.predict(X_test)

# Avaliando o desempenho do modelo com erro quadrático médio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio (MSE): {mse}")

# Exibindo previsões
print(f"Previsões do modelo: {y_pred}")
# Importando a biblioteca para fazer requisições à API
import requests  

# URL da API CoinGecko para obter preços de criptomoedas
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd"

# Fazendo a requisição para obter os dados
response = requests.get(url)

# Convertendo a resposta para JSON
if response.status_code == 200:
    data = response.json()
    print("Dados das criptomoedas:", data)
else:
    print("Erro ao acessar a API:", response.status_code)
