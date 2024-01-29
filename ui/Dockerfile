FROM python:3.9
COPY requirements.txt app/requirements.txt
WORKDIR /app
RUN apt-get update
RUN pip3 install -r requirements.txt
COPY . /app
EXPOSE 8039
CMD ["streamlit", "run", "app.py","--server.port","8039"]