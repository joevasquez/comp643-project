import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from wordcloud import WordCloud

def create_wordcloud(title, text):
    # Creating word_cloud with text as argument
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    
    # Create file name and export to /images directory
    file_name = "images/" + title + ".png"
    word_cloud.to_file(file_name)