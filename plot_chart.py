import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_chart(title, data):
    data.sort(key=lambda x: x[1], reverse=True) 

    # save the names and their respective scores separately
    # reverse the tuples to go from most frequent to least frequent 
    people = zip(*data)[0]
    score = zip(*data)[1]
    x_pos = np.arange(len(people)) 

    # calculate slope and intercept for the linear trend line
    #slope, intercept = np.polyfit(x_pos, score, 1)
    #trendline = intercept + (slope * x_pos)

    #plt.plot(x_pos, trendline, color='red', linestyle='--')    
    plt.bar(x_pos, score,align='center')
    plt.xticks(x_pos, people) 

    name = title + " Entity Frequencies"
    location = "images/" + title + "freq.png"

    plt.ylabel(name)
    plt.savefig(location)