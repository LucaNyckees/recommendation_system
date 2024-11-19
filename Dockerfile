FROM python:3.10
COPY ./ /app
WORKDIR /app

RUN apt-get update
RUN apt-get -y install make

# Install dependencies:
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

CMD [ "scripts/start.sh"]