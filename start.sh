#!/bin/bash

# Mise à jour des paquets
apt-get update

# Installation des dépendances
apt-get -y install curl
apt-get install libgomp1

streamlit run Dashboard_streamlit.py