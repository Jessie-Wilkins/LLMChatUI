FROM python:3.9.18

RUN pip install langchain ctransformers spacy faiss-cpu streamlit\
&& python -m spacy download en_core_web_sm\
&& mkdir /mistral

WORKDIR /mistral

COPY main.py main.py

CMD ["streamlit", "run", "main.py"]


