FROM python:3.11
ADD . /code
WORKDIR /code

RUN pip install --upgrade pip \
    && pip install Flask==3.0.0 \
    && pip install pandas==2.1.3 \
    && pip install keras==3.1.1 \
    && pip install firebase-admin=-6.5.0 \
    && pip install scikit-learn==1.3.2 \
    && pip install tensorflow==2.16.1

EXPOSE 5000

CMD ["python", "server.py"]