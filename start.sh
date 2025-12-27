#!/usr/bin/env bash
mkdir -p ~/.streamlit

echo "[general]
email = \"\"" > ~/.streamlit/credentials.toml

echo "[server]
headless = true
enableCORS=false
port = $PORT" > ~/.streamlit/config.toml

streamlit run motion_web_app_streamlined.py --server.port=$PORT
