import pandas as pd
import matplotlib.pyplot as plt

def display_points(X, y, title=None):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0], X[:,1], c=y)
    plt.title(title)
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
    
    **Names of OUT, STAT and DEBUG files**

    Names of OUT, STAT and DEBUG files should be related to the performed
    experiment. For instance, example name of the OUT file storing the 
    results returned by optimized NBC for fname dataset with 10000 of 
    2-dimensional points (records), where minPts = 5, Eps = 10, reference 
    point r is constructed from minimal values in domains of all dimensions
    could be as follows:

        OUT_Opt-NBC_fname_D2_R10000_m5_e10_rMin.csv
    
    '''
    file_name = (
        f'./out/{file_type}_{algorithm_name}_{dataset_name}_'
        f'D{n_dimentions}_R{n_rows}_m{minPts}_e{epsilon}_rMin.csv'
    )
    df.to_csv(file_name, index=False)