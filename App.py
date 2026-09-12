import streamlit as st
import pandas as pd
import json
from prophet.serialize import model_from_json
from prophet.plot import plot_plotly


def loadModel():
    with open('modelo_O3_prophet.json', 'r') as file_in:
        modelo = model_from_json(json.load(file_in))
        return modelo

modelo = loadModel()

# Adicionando textos ao layout do Streamlit
st.title('Previsão de Níveis de Ozônio (O3) Utilizando a Biblioteca Prophet')

st.caption('''Este projeto utiliza a biblioteca Prophet para prever os níveis de ozônio em ug/m3. O modelo
           criado foi treinado com dados até o dia 05/05/2023 e possui um erro de previsão (RMSE - Erro Quadrático Médio) igual a 17.43 nos dados de teste.
           O usuário pode inserir o número de dias para os quais deseja a previsão, e o modelo gerará um gráfico
           interativo contendo as estimativas baseadas em dados históricos de concentração de O3.
           Além disso, uma tabela será exibida com os valores estimados para cada dia.''')

st.subheader('Insira o número de dias para previsão:')

dias = st.number_input('', min_value = 1, value = 1, step=1)



if 'previsao_feita' not in st.session_state:
    st.session_state['previsao_feita'] = False
    st.session_state['dados_previsao'] = None

if st.button('Prever'):
    st.session_state.previsao_feita = True
    futuro = modelo.make_future_dataframe(periods = dias, freq = 'D')
    previsao = modelo.predict(futuro)
    st.session_state['dados_previsao'] = previsao

# código omitido

previsao = st.session_state['dados_previsao']
tabela_previsao = previsao[['ds', 'yhat']].tail(dias)
tabela_previsao.columns = ['Data (Dia/Mês/Ano)', 'O3 (ug/m³)']
tabela_previsao['Data (Dia/Mês/Ano)'] = tabela_previsao['Data (Dia/Mês/Ano)'].dt.strftime('%d-%m-%')
tabela_previsao['O3 (ug/m³)'] = tabela_previsao['O3 (ug/m³)'].round(2)
tabela_previsao.reset_index(drop = True, inplace = True)
st.write('Tabela contendo previsões de O3 (ug/m³) para os próximos {} dias'.format(dias))
st.dataframe(tabela_previsao, height=300)

csv = tabela_previsao.to_csv(index = False)
st.download_button('Baixar tabela como CSV', data = csv, file_name='prevOzonio.csv')