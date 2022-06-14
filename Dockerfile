# FROM python:3.8.6-buster

# COPY requirements.txt /requirements.txt
# COPY app.py /app.py

# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# CMD streamlit run app.py --port $PORT


FROM python:3.8.6-buster
EXPOSE 8080
COPY requirements.txt /requirements.txt
COPY app.py /app.py
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD streamlit run app.py --server.port 8080 --server.enableCORS false app.py
