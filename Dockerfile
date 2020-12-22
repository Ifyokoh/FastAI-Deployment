FROM python:3.6

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY server server/

EXPOSE 5000

CMD ["python", "server/main.py"]