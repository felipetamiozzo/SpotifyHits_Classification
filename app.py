import streamlit as st
import pandas as pd
import joblib
import os

#Configuração da Página
st.set_page_config(
    page_title="Previsor de Hits do Spotify",
    page_icon="🎵",
    layout="wide"
)

#Carregamento do modelo
# Usamos @st.cache_resource para que o modelo seja carregado apenas uma vez.
@st.cache_resource
def carregar_modelo():
    """Carrega o pipeline de modelo salvo"""
    
    caminho_modelo = os.path.join('models', 'spotify_model_pipeline.pkl')
    try:
        modelo = joblib.load(caminho_modelo)
        return modelo
    except FileNotFoundError:
        st.error(f"Erro: Arquivo do modelo não encontrado em {caminho_modelo}")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

modelo = carregar_modelo()

#Interface do usuário (Barra Lateral)

st.sidebar.title("Previsor de Hits do Spotify")
st.sidebar.header("Insira as características da música:")

def pegar_input_usuario():
    """Cria os sliders e inputs na barra lateral para coletar os dados."""
    
    #Sliders para features normalizadas (0.0 a 1.0)
    danceability = st.sidebar.slider("Dançabilidade (Danceability)", 0.0, 1.0, 0.75, 0.01)
    energy = st.sidebar.slider("Energia (Energy)", 0.0, 1.0, 0.8, 0.01)
    acousticness = st.sidebar.slider("Acústica (Acousticness)", 0.0, 1.0, 0.1, 0.01)
    instrumentalness = st.sidebar.slider("Instrumentalidade (Instrumentalness)", 0.0, 1.0, 0.0, 0.01)
    valence = st.sidebar.slider("Valência (Positividade)", 0.0, 1.0, 0.6, 0.01)
    
    #Inputs numéricos para outras features
    loudness = st.sidebar.number_input("Volume (Loudness, em dB)", -60.0, 5.0, -5.5, 0.1)
    duration_ms = st.sidebar.number_input("Duração (em ms)", 30000, 1000000, 210000, 1000)
    chorus_hit = st.sidebar.number_input("Início do Refrão (Chorus Hit)", 0.0, 300.0, 40.5, 0.1)
    
    #Selectbox/Inputs para features categóricas/discretas
    key = st.sidebar.selectbox("Tom (Key)", list(range(12)), index=5)
    mode = st.sidebar.selectbox("Modo (Mode - 1: Maior, 0: Menor)", [0, 1], index=1)
    time_signature = st.sidebar.selectbox("Compasso (Time Signature)", [1, 2, 3, 4, 5], index=3)
    sections = st.sidebar.number_input("Nº de Seções (Sections)", 1, 50, 10, 1)
    
    #A feature que você criou
    is_vocal_track = st.sidebar.selectbox("É uma faixa vocal? (is_vocal_track)", [0, 1], index=1)

    #Coleta os dados em um dicionário
    dados = {
        'danceability': danceability,
        'energy': energy,
        'key': key,
        'loudness': loudness,
        'mode': mode,
        'acousticness': acousticness,
        'instrumentalness': instrumentalness,
        'valence': valence,
        'duration_ms': duration_ms,
        'time_signature': time_signature,
        'chorus_hit': chorus_hit,
        'sections': sections,
        'is_vocal_track': is_vocal_track
    }
    
    #Converte o dicionário em um DataFrame do Pandas
    #O pipeline espera as colunas na MESMA ORDEM do X_train
    colunas_ordenadas = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'acousticness',
        'instrumentalness', 'valence', 'duration_ms', 'time_signature',
        'chorus_hit', 'sections', 'is_vocal_track'
    ]
    
    features = pd.DataFrame(dados, index=[0])
    return features[colunas_ordenadas] #Garante a ordem correta

#Página Principal
st.header("Seu Modelo em Ação")
st.write("Use a barra lateral à esquerda para ajustar os parâmetros da música e veja a previsão do modelo em tempo real.")

#Pega os dados da barra lateral
input_df = pegar_input_usuario()

#Mostra os dados inseridos (opcional)
st.subheader("Características da Música Inserida:")
st.dataframe(input_df, hide_index=True)

#Previsão e Exibição do Resultado
if modelo is not None:
    if st.sidebar.button("Prever Sucesso!", type="primary"):
        
        #Faz a previsão (o pipeline cuida da normalização automaticamente)
        try:
            previsao = modelo.predict(input_df)
            probabilidade = modelo.predict_proba(input_df)

            prob_hit = probabilidade[0][1]
            prob_flop = probabilidade[0][0]

            st.subheader("Resultado da Previsão:")

            if previsao[0] == 1:
                st.success(f"**É UM HIT!** 🚀")
                st.progress(prob_hit)
                st.markdown(f"O modelo tem **{prob_hit*100:.2f}%** de certeza de que esta música é um Hit.")
            else:
                st.error(f"**É UM FLOP.** 💔")
                st.progress(prob_flop)
                st.markdown(f"O modelo tem **{prob_flop*100:.2f}%** de certeza de que esta música é um Flop.")
        
        except Exception as e:
            st.error(f"Ocorreu um erro durante a previsão: {e}")
else:
    st.error("Modelo não carregado. Verifique o caminho 'models/spotify_model_pipeline.pkl'")