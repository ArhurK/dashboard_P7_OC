#!/bin/bash

# Mise à jour des paquets
apt-get update

# Installation des dépendances
apt-get -y install curl
apt-get install libgomp1

gunicorn -w 2 -k uvicorn.workers.UvicornWorker vs_code_api:app