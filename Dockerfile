FROM python:3.10-slim
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip
EXPOSE 8050

COPY ./baseball_data/ /app/baseball_data/
COPY ./political_data/ /app/political_data/
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY ./dash_app/ /app/dash_app/

WORKDIR /app/dash_app
ENTRYPOINT ["gunicorn", "--config", "gunicorn_config.py", "index:server"]
