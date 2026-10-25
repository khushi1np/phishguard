# PhishGuard 🛡️

A phishing URL detector built with Python and Streamlit.

## Features
- Rule-based checks: HTTPS, @ symbol, URL length, IP address, subdomains, suspicious words
- Risk score out of 100 with Safe / Suspicious / Dangerous result
- Machine learning prediction (Random Forest) with confidence
- Check history and CSV download

## Run locally
pip install -r requirements.txt
python -m streamlit run app.py

## Tech
Python, Streamlit, scikit-learn, pandas
