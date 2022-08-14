import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_chart(name, file_name, data):
    data.sort(key=lambda x: x[1], reverse=True) 

    # save the names and their respective scores separately
    # reverse the tuples to go from most frequent to least frequent 
    people = list(zip(*data))[0][:25]
    score = list(zip(*data))[1][:25]
    x_pos = np.arange(len(people)) 
    
    plt.bar(x_pos, score,align='center')
    plt.xticks(x_pos, people) 
    plt.xticks(rotation = 45)

    plt.ylabel(name)
    plt.savefig(file_name)