@echo off
echo initializing environment...
call .venv\Scripts\activate.bat

echo running streamlit app...
streamlit run app.py

pause