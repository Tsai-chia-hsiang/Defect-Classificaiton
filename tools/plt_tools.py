from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_curves(lst:list[tuple[list, str]], plt_title:str, saveto:Path):
    
    f = plt.figure(dpi=600)
    plt.grid(True)
    
    for t, l in lst:
        plt.plot(np.arange(len(t))+1, t, label=l)
    
    plt.title(plt_title)
    plt.legend()
    
    plt.savefig(saveto)
    plt.close()
