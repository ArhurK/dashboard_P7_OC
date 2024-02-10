import streamlit as st
import requests
import json
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI 
import plotly.express as px

# Titre du Dashboard
st.title("Dashboard risque de défaut")

# Interface utilisateur Streamlit
client_id = st.text_input("Renseigner l'ID du client (6 chiffres), puis cliquer sur le bouton 'Obtenir les prédictions et Shap values'", '403414')
button_clicked = st.button('Obtenir les prédictions, Shap values et données client')

if button_clicked:
    try:
        # API Prédictions
        predict_proba_url = requests.get(url='http://127.0.0.1:8000/predict_proba', json={"index": client_id})
        response_predict_proba = predict_proba_url

        # Appel API pour obtenir les Shap values
        shap_url = requests.get(url='http://127.0.0.1:8000/shap', json={"index": client_id})
        response_shap = shap_url

        if response_predict_proba.status_code == 200 and response_shap.status_code == 200:
            #############################
            # Récupéreration des données
            #############################

            data_predict_proba = response_predict_proba.json()
            prediction_value = data_predict_proba['predictions'][0][1]

            data_shap = response_shap.json()
            shap_json_dict = json.loads(data_shap) 
            print(f'shap_json_dict : {type(shap_json_dict)}')

            shap_values = shap_json_dict["shap_values"]
            print(f'shap_values : {type(shap_values)}')

            shap_values_array = np.array(shap_values)
            print(f'shap_values_array : {type(shap_values_array)}')

            X_shap = shap_json_dict['X']
            X_shap = pd.DataFrame(X_shap)

            ###########################
            # Affichage des prédictions
            ###########################

            st.subheader(f"Risque de défault du client")
            prediction_percentage = round(prediction_value * 100, 1)
            # équivalent vert / rouge accessible aux personnes daltoniennes
            progress_color = '#3498db' if prediction_percentage <= 45 else '#e74c3c'
            # Afficher la probabilité
            st.markdown(f'<p style="color:{progress_color}; font-size:30px;">Probabilité de défaut du client: {prediction_percentage}%</p>', unsafe_allow_html=True)


            #################################
            # Affichage des données du client
            #################################

            # st.subheader("Données du client")
            # st.dataframe(X_shap)

            ################################
            # Graphiques SHAP
            ################################

            st.subheader("Shap-values : Importance des variables prédictives")

            # Graphique 1 Importance des variables

            # Création d'un DataFrame pandas avec les valeurs SHAP
            shap_df = pd.DataFrame(shap_values_array, columns=X_shap.columns)
            # Calcul de la moyenne des valeurs SHAP absolues pour chaque caractéristique
            mean_shap_values = shap_df.abs().mean()
            # Sélection des 20 caractéristiques les plus prédictives
            top_features = mean_shap_values.nlargest(20).index
            # Création d'un graphique interactif avec plotly express pour les dix caractéristiques
            fig_importance = px.bar(
                mean_shap_values[top_features],
                title="<b>Top 20 des variables prédictives par SHAP Value</b><br><sup>lecture : variables qui ont le plus d'influence sur la probabilité de faire défault </sup>",
                labels={'value': 'Mean SHAP Value', 'index': 'Feature'},
                height= 600,
                width=800
                )
            fig_importance.update_layout(xaxis_title="", yaxis_title="Importance")
            st.plotly_chart(fig_importance)

            # Graphique 2 Influence des variables

            # Création d'un graphique en barres avec Plotly Express
            fig_coef = px.bar(
                shap_df[top_features],
                # title="Influence des 20 variables les plus prédictives sur le modèle",
                title = "<b>Influence des 20 variables les plus prédictives sur le modèle</b><br><sup>lecture : un coefficient positif tire la probabilité de défault à la hausse et inversement pour une valeure négative</sup>",
                labels={'index': 'Feature', 'value': 'SHAP Value'},
                barmode='group',
                height=600, width=800
                        )

            # Mise en forme du graphique
            # fig_coef.update_layout(xaxis=dict(showgrid=True, gridcolor='WhiteSmoke', zerolinecolor='Gainsboro'),
            #                 yaxis=dict(showgrid=True, gridcolor='WhiteSmoke', zerolinecolor='Gainsboro'),
            #                 plot_bgcolor='white')
            
            # Visualisation fig_coef dans streamlit
            st.plotly_chart(fig_coef)


            #################################
            # Affichage des données du client
            #################################

            st.subheader("Données du client")
            st.dataframe(X_shap)


            # Graphique 3 Synthèse 

            # # Création d'un graphique Shap values avec Matplotlib
            # fig, ax = plt.subplots()
            # shap.summary_plot(shap_values_array, X_shap, feature_names=X_shap.columns, show=False)
            # ax.set_title("Résumé des Shap Values", fontsize=16)
            # plt.tight_layout()

            # # Afficher le graphique avec Streamlit
            # # st.subheader("Shap-values : Importance des variables prédictives")
            # st.pyplot(fig)

            
        else:
            st.error(f"Erreur lors de la récupération des données. Predict Proba status: {response_predict_proba.status_code}, Shap status: {response_shap.status_code}")

    except Exception as e:
        st.error(f"Erreur inattendue: {str(e)}")



