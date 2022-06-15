FROM python:3.8.6-buster

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD streamlit run app.py --server.port $PORT
