FROM python:3.10-slim
COPY ./app.py /deploy/
COPY ./data_processing_and_features.py /deploy/
COPY ./templates /deploy/templates
COPY ./requirements.txt /deploy/
COPY ./models/model_classifier.pkl /deploy/models/model_classifier.pkl
COPY ./models/tfidf.pkl /deploy/models/tfidf.pkl
WORKDIR /deploy/
RUN pip install -r requirements.txt
# Download the NLTK stopwords data
RUN python -m nltk.downloader stopwords
EXPOSE 80
ENTRYPOINT ["python", "app.py"]