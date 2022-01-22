import pandas as pd
import matplotlib.pyplot as plt

def display_points(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=y)
    plt.show()
    
def save_file(
    df, 
    file_type, 
    algorithm_name, 
    dataset_name, 
    n_dimentions, 
    n_rows, 
    minPts, 
    epsilon
):
    '''Save the DataFrame according to the required naming convention. 
    '''
    file_name = f'''
        {file_type}_{algorithm_name}_{dataset_name}_D{n_dimentions}\
        _R{n_rows}_m{minPts}_e{epsilon}_rMin.csv
    '''
    df.to_csv(file_name)