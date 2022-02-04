import pandas as pd
import matplotlib.pyplot as plt

def display_points(X, y, title=None, numerate=False):
    fig, ax = plt.subplots()
    if numerate:
        for i, txt in enumerate(range(len(X))):
            ax.annotate(i, (X[i,0], X[i,1]), size=15)
    ax.scatter(X[:,0], X[:,1], c=y)    
    plt.title(title)
    plt.show()
    
    
def get_name( 
    algorithm_name, 
    dataset_name, 
    n_dimentions, 
    n_rows, 
    minPts='', 
    epsilon='',
    similarity='',
    k='',
    ref_point=''
):
    name = (
        f'{algorithm_name}_{dataset_name}'
        f'_D{n_dimentions}_R{n_rows}'
    )
    if minPts:
        name += f'_m{minPts}' 
    if epsilon:
        name += f'_e{epsilon}'
    if k:
        name += f'_k{k}'
    if similarity:
        name += f'_{similarity.__name__}'
    if ref_point:
        name += f'_r{ref_point}'
    return name
    