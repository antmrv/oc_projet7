import streamlit as st
import pandas as pd
import pickle
import requests
import json
import ast
from streamlit_echarts import st_echarts
from st_aggrid import AgGrid
import seaborn as sns
import matplotlib.pyplot as plt

#st.set_page_config(layout="wide")

st.title("Bienvenue sur la démo de dashboard !")

# Données des clients test
pickle_in = open("test_set.pkl","rb")
data_test = pickle.load(pickle_in)

# Données sur les seuils d'acceptation
pickle_in = open("defaut_indicateurs.pkl","rb")
defaut = pickle.load(pickle_in)
defaut["seuil"] = defaut["seuil"].round(4)

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}
    data_json = {'client_id': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    """
    Configure les differents ports de l'API
    Cree une liste deroulante pour chosir l'application utilise
    """
    FastAPI_URI_pred = "https://amprojet7.herokuapp.com/predict_proba"
    FastAPI_URI_show = "https://amprojet7.herokuapp.com/show"
    FastAPI_URI_explain = "https://amprojet7.herokuapp.com/explain"

    st.title("Informations du client")
    
    
    # Sidebar
    st.sidebar.header('Paramètres')
    with st.sidebar.container():
        # Choix d'un client    
        st.sidebar.subheader('Client')
        index = st.sidebar.selectbox('Quel client souhaitez vous consulter ?', data_test['SK_ID_CURR'].iloc[:100].to_list())
    
    # Choix de seuil
    with st.sidebar.container():
        # Seuils pour lesquels les indicateurs ont été calculés
        seuils = defaut["seuil"].round(4).tolist()

        st.sidebar.subheader("Choisissez les seuils d'acceptation")
        
        # Seuil acceptation
        accept_value = st.sidebar.select_slider(
                            "Choisissez un seuil d'acceptation du dossier, le seuil conseillé est de 0.07",
                            options = seuils)
        
        # Informations sur le seuil 
        defaut_index = defaut.loc[defaut["seuil"] == accept_value].index
        tx_refus =  defaut.iloc[defaut_index, 3].values[0].round(2)
        st.sidebar.write("Taux de dossiers refusés pour ce seuil : ", tx_refus)
        tx_defaut_accepte = defaut.iloc[defaut_index, 4].values[0].round(2)
        st.sidebar.write("Taux de défaut des dossiers acceptés pour ce seuil : ", tx_defaut_accepte)
                               
        # Seuil refus
        seuils_refus = [valeur for valeur in seuils if valeur > accept_value]
        refuse_value = st.sidebar.select_slider(
         "Choisissez un seuil de refus du dossier supérieur au seuil d'acceptation automatique, le seuil conseillé est de 0.20",
                            options = seuils_refus)
        
        # Informations sur le seuil
        defaut_index = defaut.loc[defaut["seuil"] == refuse_value].index
        tx_refus =  defaut.iloc[defaut_index, 3].values[0].round(2)
        st.sidebar.write("Taux de dossiers refusés pour ce seuil :", tx_refus)
        tx_defaut_accepte = defaut.iloc[defaut_index, 4].values[0].round(2)
        st.sidebar.write("Taux de défaut des dossiers acceptés pour ce seuil : ", tx_defaut_accepte)
        
        
    # Affiche les information du client
    with st.container():
        data = index
        Affichage = request_prediction(FastAPI_URI_show,  data)
        AgGrid(pd.DataFrame(ast.literal_eval(Affichage)))
        
   
    # Probabilité de défaut pour un client (prediction du modèle)
    with st.container():    
        data = index
        pred = request_prediction(FastAPI_URI_pred, data)

        option = {
            "tooltip": {
                "formatter": '{a} <br/>{b} : {c}%'
            },
            "series": [{
                "name": 'Proba',
                "type": 'gauge',
                "axisLine": {
                    "lineStyle": {
                        "width": 30,
                        "color": [
                                [accept_value, '#2FEB00'], # seuils intermédiaires
                                [refuse_value, '#FF8114'],
                                [100, '#FF0000']
                            ]
                        }
                    },
                 "pointer": {
                     "itemStyle": {
                     "color": 'inherit'
                        }
                 },
                "axisTick": {
                    "distance": -30,
                    "length": 8,
                    "lineStyle": {
                      "color": '#fff',
                      "width": 2
                     }
                },
                "splitLine": {
                    "distance": -30,
                    "length": 30,
                    "lineStyle": {
                      "color": '#fff',
                      "width": 4
                    }
                  },
                  "axisLabel": {
                    "color": 'inherit',
                    "distance": 40,
                    "fontSize": 20
                  },
                  "detail": {
                    "valueAnimation": 'true',
                    "formatter": '{value}',
                    "color": 'inherit'
                  },

                "data": [{
                    "value": round(float(pred['predict_proba']) * 100, 1) ,
                    "name": 'Prediction'
                }]
            }]
        };


        st_echarts(options=option,height="600px", key="1") 

        
    


   # Explicabilité avec LIME
    with st.container():
        data = index

        explain_plot = ast.literal_eval(request_prediction(FastAPI_URI_explain, data))
        df_explain_plot = pd.DataFrame(explain_plot)
       # st.write(df_explain_plot)
        df_explain_plot['positive'] = df_explain_plot[1] > 0
        fig = plt.figure(figsize=(10, 4))
        sns.barplot(data = df_explain_plot, x = 1, y = 0, hue = 'positive' , palette = 'rocket')
        plt.xlabel('Pouvoir prédictif de la variable')
        plt.ylabel('Variables')
        st.pyplot(fig=fig, clear_figure = True)
        plt.close()
        
        
    with st.container() :
        Colonne = st.selectbox('Quel variable souhaitait vous consulter ? ', data_test.drop(['SK_ID_CURR'], axis = 1).columns)
        Val_credit = float(data_test[data_test['SK_ID_CURR'] == index][Colonne])
        
        
    with st.container():
        fig = plt.figure(figsize=(10, 4))
        sns.kdeplot(data = data_test, x = Colonne, hue = 'TARGET', common_norm = True)
        plt.plot([Val_credit, Val_credit] , [0,1], 'r', linestyle = 'dashed')
        plt.title(f'Le credit predit se place de cette manière pour la variable {Colonne}')
        plt.xlabel(f'Valeur {Colonne}')
        plt.ylabel(f'Répartition {Colonne} Defaut et Sains')
        st.pyplot(fig=fig)


# streamlit run dashboard.py, conda activate p7_env, cd oc_projet7        
if __name__ == '__main__':
    main()
