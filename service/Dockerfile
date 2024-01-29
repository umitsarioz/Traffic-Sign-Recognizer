FROM python:3.9

COPY requirements.txt app/requirements.txt

WORKDIR /app
RUN apt-get update
RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 8040

CMD ["uvicorn", "fast-api:app", "--host", "0.0.0.0", "--port", "8040"]