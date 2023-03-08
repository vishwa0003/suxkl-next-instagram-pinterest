FROM tiangolo/uwsgi-nginx-flask:python3.8

COPY ./requirements.txt /app/
RUN python3.8 -m pip install -r /app/requirements.txt

COPY . .

EXPOSE 8080

CMD [ "python", "app.py" ]
