import pandas as pd
from src.datasets import Dataset
from src.utils import *
from src.clusterization_performance import *
import glob
from pathlib import Path
import json

config_path = './configs/dbscan.json'
f = open(config_path)
config = json.load(f)


for out_path, log_path in zip(glob.glob('out/dbscan*.csv'), glob.glob('out/LOG*.log')):
    # print('\n')
    # print(log_path)
    # print(out_path)
    # print('\n')
    
    name = Path(log_path).stem.replace('LOG_','')

    # lad dataset
    if len([dataset for dataset in list(config.keys()) if dataset in name]) > 0:

        # load out
        df = pd.read_csv(out_path, header=None, index_col=0)
        y_pred, state = df[1], df[2]

        dataset_name = [dataset for dataset in list(config.keys()) if dataset in name][0]

        dataset = Dataset(f'data/{dataset_name}.txt', print_info=False) 
        X, y = dataset.X, dataset.y
        
        logs = pd.read_csv(
            f'out/LOG_{name}.log',
            names=['time [ms]', 'operation', 'point_id', 'value', 'string'],
            index_col=False
        )

        # OUT
        mask = logs['operation'] == 'similarity_calculation'
        similarity_calculation = logs[mask].groupby('point_id').sum().reset_index()
        similarity_calculation.rename(columns={'value': '# of distance/similarity calculations'}, inplace=True)

        out = pd.DataFrame({
            'point_id': np.arange(dataset.n_rows),
            'x': dataset.X[:, 0], 
            'y': dataset.X[:, 1],
            'point_type': state, # 1 - core, 0 - border, -1 - noise
            'CId': y_pred # clusters: cluster identifier or -1 in the case of noise points
        })

        out = out.merge(similarity_calculation, on='point_id')
        out.to_csv(f'./out/OUT_{name}.csv', index=False)
        
        # STAT
        params = config[dataset_name]['params_dbscan']

        score = evaluate(y_pred, dataset.y, dataset.X)
        stat = pd.DataFrame({
            'name of the input fil': dataset.name,
            '# of dimensions of a point': dataset.n_dimentions,
            '# of points in the input file': dataset.n_rows,
            'epsilon': params['epsilon'] if 'epsilon' in params else '',
            'minPts': params['minPts'] if 'minPts' in params else '',
            'k':  params['k'] if 'k' in params else '',
            # 'similarity': params['similarity'],
            'values of dimensions of a reference point': '[0,1]',
            'reading the input file [ms]': logs.loc[logs['operation'] == 'reading_data', 'value'].values[0],
            'normalization of vectors [ms]': "",
            "Eps-neighborhood timer [ms]": logs.loc[logs['operation'] == 'Eps_time', 'value'].sum(),
            'Clustering timer [ms]': (
                logs.loc[logs['operation'] == 'stop log', 'time [ms]'].values[0] - 
                logs.loc[logs['operation'] == 'start log', 'time [ms]'].values[0]
            ),
            # 'saving results to OUT time [ms]': logs.loc[logs['operation'] == 'writing_data', 'value'].values[0] ,
            "dist_to_ref_point_time [ms]": logs[logs['operation'] == 'dist_to_ref_point_time']['value'].sum(),
            'total runtime [ms]': (
                logs.loc[logs['operation'] == 'writing_data', 'time [ms]'].values[0] - 
                logs.loc[logs['operation'] == 'reading_data', 'time [ms]'].values[0]
            ),
            "sorting_dist_time [ms]": logs[logs['operation'] == 'sorting_pessimistic_est_time']['value'].sum(),
            "tiknn_time [ms]": logs[logs['operation'] == 'knn_time']['value'].sum(),
            "tirnn_time [ms]": logs[logs['operation'] == 'rnn_time']['value'].sum(),
            '# of discovered clusters': sum(out['CId'] > -1),
            '# of discovered noise points': sum(out['point_type'] == -1),
            '# of discovered core points': sum(out['point_type'] == 1),
            '# of discovered border points': sum(out['point_type'] == 0),
            'avg # of calculations of distance/similarity': out['# of distance/similarity calculations'].mean(),
            '|TP|': score['TP'],
            '|TN|': score['TN'],
            '# of pairs of points': len(y),
            'RAND': score['rand_score'],
            'Purity': score['purity'],
            'Silhouette coefficient': score['silhouette_score_euclidean'],
            # 'Davies Bouldin': score['davies_bouldin_score']
        }, index=['values']).T
        
        stat.to_csv(f'./out/STAT_{name}.csv', index=True)
        
        # DEBUG
        mask1 = logs['operation'] == '|knn_neighbors|'
        mask2 = logs['operation'] == 'knn_neighbors_id'
        mask3 = logs['operation'] == '|rnn_neighbors|'
        mask4 = logs['operation'] == 'rnn_neighbors_id'

        debug1 = (
            logs[mask1]
            .pivot_table(
                index=['point_id'], 
                columns=['operation'], 
                values= 'value', 
            )
        )
        debug1.reset_index(col_level=1, inplace=True)
        debug1.columns.name = None

        debug2 = (
            logs[mask2]
            .pivot_table(
                index=['point_id'], 
                columns=['operation'], 
                values= 'string', 
                aggfunc=lambda x: ' '.join(x)
        ))   
        debug2.reset_index(col_level=1, inplace=True)
        debug2.columns.name = None

        debug3 = (
            logs[mask3]
            .pivot_table(
                index=['point_id'], 
                columns=['operation'], 
                values= 'value', 
            )
        )
        debug3.reset_index(col_level=1, inplace=True)
        debug3.columns.name = None

        debug4 = (
            logs[mask4]
            .fillna('')
            .pivot_table(
                index=['point_id'], 
                columns=['operation'], 
                values= 'string', 
                aggfunc=lambda x: ' '.join(x)
        ))   
        debug4.reset_index(col_level=1, inplace=True)
        debug4.columns.name = None


        debug = debug1.merge(debug2, on='point_id').merge(debug3, on='point_id').merge(debug4, on='point_id')
        debug.to_csv(f'./out/DEBUG_{name}.csv', index=False)

        # Plots
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.scatter(X[:,0], X[:,1], c=y_pred)    
        ax.set_title(name)
        fig.savefig(f"./img/{name}.jpeg")
        plt.close(fig)  
