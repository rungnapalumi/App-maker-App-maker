#!/usr/bin/env bash
mkdir -p ~/.streamlit

echo "\
[general]\n\
email = \"\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

streamlit run motion_web_app_streamlined.py --server.port=$PORT
