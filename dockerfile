FROM python:3.10-slim

EXPOSE 8501

WORKDIR /app

RUN /usr/local/bin/python -m pip install --upgrade pip

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "Indice.py", "--server.port=8501", "--server.address=0.0.0.0"]
