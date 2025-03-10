from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Carregar os dados salvos
df = pd.read_csv('historico_precos_bitcoin.csv')

# Criar a variável alvo (se o preço do próximo dia é maior ou menor que o atual)
df['target'] = np.where(df['price'].shift(-1) > df['price'], 1, 0)

# Remover a última linha, pois ela não tem alvo
df = df[:-1]

# Definir as variáveis independentes e dependentes
X = df[['price']]
y = df['target']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")
