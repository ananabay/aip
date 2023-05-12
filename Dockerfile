FROM python:3.9-slim-buster

WORKDIR /aip

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
# RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
