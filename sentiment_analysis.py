import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline

# 1. Configuração do Modelo de IA (Deep Learning - BERT Multilingual)
# Este modelo é excelente para Português e identifica de 1 a 5 estrelas
print("Carregando modelo de IA...")
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# 2. Dados de Exemplo (Simulando uma Agência de Marketing)
data = {
    'data': ['2024-01-10', '2024-01-11', '2024-01-11', '2024-01-12', '2024-01-12'],
    'cliente': ['TechStore', 'TechStore', 'BioHacker', 'BioHacker', 'TechStore'],
    'feedback': [
        "O suporte foi rápido e resolveu meu problema com os anúncios.",
        "Não gostei da nova arte da campanha, as cores estão estranhas.",
        "A estratégia de SEO está dando muito certo, dobramos as visitas!",
        "O relatório mensal atrasou e os dados parecem confusos.",
        "Equipe fantástica, recomendo muito o trabalho de tráfego pago."
    ]
}

df = pd.DataFrame(data)

# 3. Processamento de IA
def get_ai_sentiment(text):
    result = sentiment_pipeline(text)[0]
    # O modelo retorna '1 star' até '5 stars'
    stars = int(result['label'].split()[0])
    
    if stars <= 2: return 'Negativo'
    elif stars == 3: return 'Neutro'
    else: return 'Positivo'

print("Analisando sentimentos...")
df['sentimento'] = df['feedback'].apply(get_ai_sentiment)

# 4. Cálculo de KPI de Negócio: Proporção de Detratores
total_negativos = len(df[df['sentimento'] == 'Negativo'])
churn_risk = (total_negativos / len(df)) * 100

print(f"\n--- RELATÓRIO EXECUTIVO ---")
print(f"Risco de Churn Estimado: {churn_risk:.2f}%")
print(df[['cliente', 'feedback', 'sentimento']])

# 5. Visualização Profissional para Dashboard
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
plot = sns.countplot(x='sentimento', data=df, palette={'Positivo': 'green', 'Negativo': 'red', 'Neutro': 'gray'})
plt.title('Análise de Sentimento: Visão Geral da Agência')
plt.ylabel('Quantidade de Feedbacks')
plt.xlabel('Categoria de Sentimento')
plt.show()
