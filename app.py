# Importar as bibliotecas necessárias
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go  # <<< LINHA ADICIONADA PARA CORRIGIR O ERRO
import time

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Previsão de Pedidos",
    page_icon="📈",
    layout="wide"
)

# --- TÍTULO E INTRODUÇÃO ---
st.title("📈 Previsão de Pedidos Jumbo CDP")
st.write("""
Esta aplicação utiliza o modelo Prophet para gerar previsões de séries temporais.
Use o controle na barra lateral para selecionar quantos meses você deseja prever.
""")


# --- FUNÇÃO DE TREINAMENTO DO MODELO (COM CACHE) ---
# @st.cache_data é um "decorador" mágico do Streamlit.
# Ele garante que o modelo seja treinado apenas uma vez, e não a cada interação do usuário.
# Isso torna o app MUITO mais rápido.
@st.cache_data
def treinar_modelo():
    """
    Função para carregar os dados e treinar o modelo Prophet.
    O resultado fica em cache para performance.
    """
    # Lista de dados explícita que criamos anteriormente
    dados_lista = [
        {'ds': '2022-01-01', 'y': 802}, {'ds': '2022-02-01', 'y': 611},
        {'ds': '2022-03-01', 'y': 722}, {'ds': '2022-04-01', 'y': 705},
        {'ds': '2022-05-01', 'y': 896}, {'ds': '2022-06-01', 'y': 816},
        {'ds': '2022-07-01', 'y': 689}, {'ds': '2022-08-01', 'y': 812},
        {'ds': '2022-09-01', 'y': 692}, {'ds': '2022-10-01', 'y': 586},
        {'ds': '2022-11-01', 'y': 544}, {'ds': '2022-12-01', 'y': 509},
        {'ds': '2023-01-01', 'y': 660}, {'ds': '2023-02-01', 'y': 518},
        {'ds': '2023-03-01', 'y': 608}, {'ds': '2023-04-01', 'y': 540},
        {'ds': '2023-05-01', 'y': 713}, {'ds': '2023-06-01', 'y': 640},
        {'ds': '2023-07-01', 'y': 627}, {'ds': '2023-08-01', 'y': 575},
        {'ds': '2023-09-01', 'y': 544}, {'ds': '2023-10-01', 'y': 584},
        {'ds': '2023-11-01', 'y': 530}, {'ds': '2023-12-01', 'y': 548},
        {'ds': '2024-01-01', 'y': 660}, {'ds': '2024-02-01', 'y': 515},
        {'ds': '2024-03-01', 'y': 551}, {'ds': '2024-04-01', 'y': 687},
        {'ds': '2024-05-01', 'y': 731}, {'ds': '2024-06-01', 'y': 706},
        {'ds': '2024-07-01', 'y': 794}, {'ds': '2024-08-01', 'y': 687},
        {'ds': '2024-09-01', 'y': 744}, {'ds': '2024-10-01', 'y': 797},
        {'ds': '2024-11-01', 'y': 735}, {'ds': '2024-12-01', 'y': 729},
        {'ds': '2025-01-01', 'y': 984}, {'ds': '2025-02-01', 'y': 814},
        {'ds': '2025-03-01', 'y': 821}, {'ds': '2025-04-01', 'y': 926},
        {'ds': '2025-05-01', 'y': 980}, {'ds': '2025-06-01', 'y': 976},
        {'ds': '2025-07-01', 'y': 929}
    ]
    df = pd.DataFrame(dados_lista)
    df['ds'] = pd.to_datetime(df['ds'])

    # Treina o modelo
    modelo = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=False,
                     daily_seasonality=False)
    modelo.fit(df)

    return modelo


# --- CARREGANDO O MODELO (E MOSTRANDO UMA MENSAGEM DE PROGRESSO) ---
with st.spinner('Treinando o modelo... Isso pode levar um momento.'):
    modelo = treinar_modelo()
st.success('Modelo treinado com sucesso!', icon="✅")

# --- BARRA LATERAL COM CONTROLES ---
st.sidebar.header('Defina os Parâmetros')
meses_previsao = st.sidebar.slider('Meses para prever', 1, 24, 12)

# --- GERAR PREVISÕES ---
futuro = modelo.make_future_dataframe(periods=meses_previsao, freq='M')
previsao = modelo.predict(futuro)

# --- EXIBIR RESULTADOS ---
st.subheader(f'Previsão para os próximos {meses_previsao} meses')

# Usamos as funções de plotagem do Plotly para gráficos interativos
fig1 = plot_plotly(modelo, previsao)
fig1.update_layout(title_text='Previsão de Pedidos', xaxis_title='Data', yaxis_title='Pedidos')
st.plotly_chart(fig1, use_container_width=True)

st.subheader('Dados da Previsão')
st.write(previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(meses_previsao))

st.subheader('Componentes da Previsão')
fig2 = plot_components_plotly(modelo, previsao)
st.plotly_chart(fig2, use_container_width=True)
