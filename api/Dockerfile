FROM python:3.12-trixie

RUN apt update -y &&  apt upgrade -y

WORKDIR /usr/api

# Create virtual enviroment for the project
COPY . requirements.txt
RUN python -m venv api_venv
RUN /bin/bash api_venv/bin/activate
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["fastapi", "run", "main.py"]