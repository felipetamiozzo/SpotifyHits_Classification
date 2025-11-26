import streamlit as st
import pandas as pd
import joblib
import os

# Configuração Inicial 
# Define o título e ícone da aba do navegador
st.set_page_config(
    page_title="Previsor de Hits",
    page_icon="🎵"
)

# Título principal na página
st.title("🎵 Vai ser Hit ou Flop?")
st.write("Descubra se sua música tem potencial para ser um sucesso no Spotify!")

# Carregando o Modelo 
# Função spara carregar o arquivo do modelo
# Usar cache para não carregar toda vez que mexer num botão
@st.cache_resource
def carregar_meu_modelo():
    try:
        # Tenta carregar da pasta 'models'
        return joblib.load('models/spotify_model_pipeline.pkl')
    except:
        # Se não achar, tenta no diretório atual (comum em alguns deploys)
        try:
            return joblib.load('spotify_model_pipeline.pkl')
        except:
            return None

modelo = carregar_meu_modelo()

# Caso o modelo não carregar, parar tudo e avisar
if modelo is None:
    st.error("⚠️ Erro: Não encontrei o arquivo do modelo. Verifique a pasta.")
    st.stop()


# Barra Lateral (Configurações)
st.sidebar.header("🎛️ Características da Música")

# Sliders para valores de 0 a 1
danceability = st.sidebar.slider("Dançabilidade (0 a 1)", 0.0, 1.0, 0.7)
energy = st.sidebar.slider("Energia (0 a 1)", 0.0, 1.0, 0.8)
valence = st.sidebar.slider("Positividade (0 a 1)", 0.0, 1.0, 0.6)
acousticness = st.sidebar.slider("Acústica (0 a 1)", 0.0, 1.0, 0.1)
instrumentalness = st.sidebar.slider("Instrumental (0 a 1)", 0.0, 1.0, 0.0)

st.sidebar.markdown("---") # Linha divisória

# Inputs numéricos
loudness = st.sidebar.number_input("Volume (dB - Negativo é mais baixo)", value=-5.0)
duration_ms = st.sidebar.number_input("Duração (em milissegundos)", value=200000)
chorus_hit = st.sidebar.number_input("Segundos até o refrão", value=30.0)

# Valores padrão para o que não é tão importante
key = 5
mode = 1
time_signature = 4
sections = 10

# Lógica automática para Vocal
# Se instrumental for muito baixo (menor que 0.01), consideramos que tem vocal
if instrumentalness < 0.01:
    is_vocal_track = 1
else:
    is_vocal_track = 0

#  Botão e Previsão 
if st.button("Analisar Música 🚀", use_container_width=True):
    
    # 1. Organizar os dados
    dados_musica = {
        'danceability': [danceability],
        'energy': [energy],
        'key': [key],
        'loudness': [loudness],
        'mode': [mode],
        'acousticness': [acousticness],
        'instrumentalness': [instrumentalness],
        'valence': [valence],
        'duration_ms': [duration_ms],
        'time_signature': [time_signature],
        'chorus_hit': [chorus_hit],
        'sections': [sections],
        'is_vocal_track': [is_vocal_track]
    }

    # 2. Criar a tabela (DataFrame)
    df_input = pd.DataFrame(dados_musica)
    
    
    # O modelo precisa receber exatamente nessa ordem 
    colunas_corretas = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'acousticness', 'instrumentalness', 'valence', 'duration_ms', 
        'time_signature', 'chorus_hit', 'sections', 'is_vocal_track'
    ]
    
    # Reorganiza as colunas para garantir
    df_input = df_input[colunas_corretas]

    # 3. Mostrando os dados na tela  (Debug visual)
    st.write("🔍 Dados enviados para o modelo:")
    st.dataframe(df_input)

    # 4. Fazendo a previsão
    if modelo is not None:
        try:
            # Pega a classe (0 ou 1)
            resultado = modelo.predict(df_input)[0]
            
            # Pega a probabilidade (ex: 0.85)
            proba = modelo.predict_proba(df_input)[0]
            chance_hit = proba[1] # Probabilidade de ser classe 1 (Hit)
            
            st.markdown("---")
            st.subheader("📊 Resultado da Análise")
            
            # Mostra a "nota" exata que o modelo deu
            st.write(f"**Probabilidade de Sucesso calculada:** {chance_hit * 100:.2f}%")
            st.progress(chance_hit)
            
            if resultado == 1:
                st.success("### 🚀 Previsão: VAI SER HIT!")
                st.balloons()
            else:
                st.error("### 📉 Previsão: Provável FLOP")
                st.info("Dica: Tente aumentar a Dançabilidade e o Volume, e zerar o Instrumental.")
                
        except Exception as e:
             st.error(f"Ocorreu um erro técnico na previsão: {e}")