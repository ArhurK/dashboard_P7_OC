#!/bin/bash

# # Mise à jour des paquets
apt-get update

# # Installation des dépendances
apt-get -y install curl
apt-get install libgomp1

# lancement de l'application
# python -m streamlit run Dashboard_streamlit.py
python -m streamlit run Dashboard_streamlit.py --server.port 8000 --server.address 0.0.0.0
