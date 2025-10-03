#!/bin/bash

# Activate conda environment and run streamlit with the correct Python
source ~/miniforge3/bin/activate base
python -m streamlit run motion_web_app_streamlined.py --server.port 8507 