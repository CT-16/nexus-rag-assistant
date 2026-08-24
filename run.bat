@echo off
title Starting NexusRAG Knowledge Assistant...
cd /d "%~dp0"
call venv\Scripts\activate
streamlit run app.py
pause