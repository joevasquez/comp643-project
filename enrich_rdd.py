from textblob import TextBlob
from bs4 import BeautifulSoup
import requests
import re
import spacy
nlp = spacy.load("en_core_web_sm")

def url_to_string(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    for script in soup(["script", "style", 'aside']):
        script.extract()
    return " ".join(re.split(r'[\n\t]+', soup.get_text()))
