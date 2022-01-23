import numpy as np
import logging
import time 

logging.basicConfig(
    filemode = 'w+',
    level=logging.INFO, 
    filename='./out/dbscan_log.log', 
    format='%(message)s'
)
# logging.info(calculation_Epsneighborhood_time)


def get_neighbors(X, current_index, epsilon, similarity):
    '''
    X - dataset with point coordinates,
    current_index - index of given point,
    epsilon - max similarity or distance to given point,
    similarity - metric of similarity or distance bewteen two points
    '''
    neighbor_indices = []
    for neighbor_index, neighbor in enumerate(X):
        if similarity(neighbor, X[current_index]) <= epsilon:
            neighbor_indices.append(neighbor_index)
    return neighbor_indices

def dbscan(X, epsilon, minPts, similarity):
    '''
    X - dataset with point coordinates,
    epsilon - max similarity or distance to given point,
    minPts - minimum number of points that create cluster,
    similarity - metric of similarity or distance bewteen two points
    '''
    timer0 = time.time() 
    # each data point can be in one of 3 stages
    NOT_VISITED = -1 # not visited point
    VISTED = 0 # non-core point
    CLUSTERED = 1 # core point
    
    # initial setup
    n = X.shape[0]
    cluster = np.array([-1] * n) # cluster register
    state = np.array([NOT_VISITED] * n) # state register
    cluster_id = 1
    number_of_calc = {} # number of distance/similarity calculations

    def search(current_index, cluster_id, epsilon, minPts, similarity):
        # calculation of Eps-neighborhood for current_index
        timer1 = time.time() 
        neighbor_indices = get_neighbors(X, current_index, epsilon, similarity)
        logging.info(f'Eps,{current_index}, {time.time() - timer1}')
        number_of_calc[current_index] = len(neighbor_indices)
        if len(neighbor_indices) >= minPts:
            state[current_index] = CLUSTERED
            cluster[current_index] = cluster_id
            for neighbor_index in neighbor_indices:
                if state[neighbor_index] == NOT_VISITED:
                    state[neighbor_index] = CLUSTERED
                    cluster[neighbor_index] = cluster_id
                    search(neighbor_index, cluster_id, epsilon, minPts, similarity)
        else:
            state[current_index] = VISTED

    while NOT_VISITED in state:
        not_visited_ids = np.where(state==NOT_VISITED)[0]
        search(not_visited_ids[0], cluster_id, epsilon, minPts, similarity)
        cluster_id += 1
    logging.info(f'cluster,, {time.time() - timer0}')
    
    return cluster, state, number_of_calc