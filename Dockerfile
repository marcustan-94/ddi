# FROM python:3.8.6-buster

# WORKDIR /app

# COPY requirements.txt ./requirements.txt

# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# EXPOSE 8501

# COPY . /app

# ENTRYPOINT [ "streamlit", "run" ]

# CMD ["app.py"]

FROM python:3.8.6-buster

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

CMD streamlit run app.py
