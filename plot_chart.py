from cgitb import small
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_chart(name, file_name, data):
    
    data.sort(key=lambda x: x[1], reverse=True) 
    people = list(zip(*data))[0][:20]
    score = list(zip(*data))[1][:20]
    x_pos = np.arange(len(people)) 

    print("Plot name: ", name)
    print("People: ", people)
    print("Scores: ", score)
    
    plt.bar(x_pos, score, align='center')
    plt.xticks(x_pos, people) 

    plt.xticks(
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='small' 
    )
    
    plt.subplots_adjust(bottom=0.2)
    plt.draw()

    plt.ylabel(name)
    plt.savefig(file_name)
    plt.show()

    plt.close()