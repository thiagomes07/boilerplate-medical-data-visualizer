# Importando as bibliotecas necessárias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Passo 1: Importando o dataset
df = pd.read_csv('medical_examination.csv')

# Passo 2: Criando a coluna 'overweight' baseada no cálculo do IMC (BMI)
# Fórmula: BMI = peso(kg) / altura(m)^2
# Se BMI > 25, a pessoa é considerada acima do peso (1), caso contrário (0)
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

# Passo 3: Normalizando os valores de colesterol e glicose
# Onde colesterol e glicose = 1, o valor é bom (0)
# Onde colesterol e glicose > 1, o valor é ruim (1)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# Passo 4: Função para desenhar o gráfico categórico
def draw_cat_plot():
    # Passo 5: Convertendo o DataFrame para o formato longo (long format)
    # Usamos pd.melt para transformar as colunas em variáveis categóricas
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Passo 6: Agrupando os dados por 'cardio', 'variable' e 'value'
    # E contando as ocorrências de cada categoria
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size().rename(columns={'size': 'total'})

    # Passo 7: Desenhando o gráfico usando seaborn's catplot
    # catplot() cria gráficos categóricos. Aqui, dividimos o gráfico por 'cardio' (colunas)
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', kind='bar', data=df_cat).fig

    # Passo 8: Salvando a figura gerada
    fig.savefig('catplot.png')
    return fig

# Passo 10: Função para desenhar o heatmap
def draw_heat_map():
    # Passo 11: Limpando os dados
    # Filtrando dados que atendem os critérios de pressão sanguínea e percentis de altura/peso
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Passo 12: Calculando a matriz de correlação
    corr = df_heat.corr()

    # Passo 13: Gerando uma máscara para ocultar a parte superior da matriz de correlação (triângulo superior)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Passo 14: Configurando a figura do matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # Passo 15: Desenhando o heatmap utilizando seaborn
    sns.heatmap(corr, annot=True, mask=mask, fmt='.1f', center=0, square=True, linewidths=.5, cbar_kws={'shrink': .5}, ax=ax)

    # Passo 16: Salvando a figura gerada
    fig.savefig('heatmap.png')
    return fig
