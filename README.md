# DBSCAN implementation
## Datasets to be used in the experiments 
- the toy dataset used in the class to present the execution of the NBC clustering algorithm (slide 51)
- dim512 from http://cs.joensuu.fi/sipu/datasets/
- complex9 from https://github.com/deric/clustering-benchmark/tree/master/src/main/resources/datasets/artificial
- cluto-t7-10k from https://github.com/deric/clustering-benchmark/tree/master/src/main/resources/datasets/artificial
- letter from https://github.com/deric/clustering-benchmark/tree/master/src/main/resources/datasets/real-world



## Results to be returned 

The results returned by a clustering algorithm for a given dataset and parameter values should be saved in in 3 files:

(1) **OUT** - an output file that contains the following information on a separate line for each point in the dataset:
point id, x, y, ...., \# of distance/similarity calculations, point type, CId
where:
- [x] point id - the position of the point in the dataset (before possible sorting),
- [x] x, y, ... - dimension values
- [x] \# of distance/similarity calculations = \# of performed calculations of the distance/similarity of the point to other points in the dataset,
- [x] a point type is either a core point (denoted by 1), or a border point (denoted by 0), or a noise point (denoted by -1)
- [x] CId is either a cluster identifier (which is a natural number) or -1 in the case of noise points.

(2) **STAT** - a file with the following statistics:
- [x] name of the input file, 
- [x] \# of dimensions of a point, 
- [x] \# of points in the input file
- [x] values of respective parameters of the used algorithm
- [ ] values of dimensions of a reference point (for TI- versions)
- [ ] partial runtimes for each important phase of algorithms such as:
    - [x] reading the input file, 
    - [ ] calculation of distances to a reference point (for TI- versions), 
    - [x] normalization of vectors (for versions working on normalized vectors), 
    - [ ] sorting of points w.r.t. their distances to the reference point (for TI- versions), 
    - [x] ~sorting of points w.r.t. their lengths (for optimized versions using the Tanimoto similarity),~ 
    - [x] calculation of Eps-neighborhood / k+NN / kNN / R-k+NN / R-kNN etc., 
    - [x] clustering, 
    - [ ] saving results to OUT and STAT files
- [x] total runtime,
- [x] \# of discovered clusters, 
- [x] \# of discovered noise points, 
- [x] \# of discovered core points, 
- [x] \# of discovered border points - if applicable
- [x] avg \# of calculations of distance/similarity of a point to other points in the data set,
- [x] |TP|, 
- [x] |TN|, 
- [x] \# of pairs of points, 
- [x] RAND and Purity (when calculating RAND and Purity, the set of all noise points should be treated as a special type of a cluster) - calculated in the case of datasets whose real clusters are known
- [x] Silhouette coefficient (when calculating Silhouette coefficient, each noise point should be treated as a separate singleton cluster)
- [x] Davies Bouldin - calculated in the case of algorithms that do not discover noise points

(3) **DEBUG** – a file that contains important information for each point in the dataset on a separate line. For example, in the case of the NBC algorithm using the Euclidean distance and TI, the following information should be returned for each point:

    point id, maxEps, minEps, NDF, |R-k+NN|, |k+NN|, identifiers of k+NN
where:
- maxEps – Eps value calculated based on first k candidates for k+NN of the point (for TI-optimized versions)
- minEps is the minimal value of radius Eps within which real k+NN of the point was found (for TI-optimized versions).

## Names of OUT, STAT and DEBUG files 
Names of OUT, STAT and DEBUG files should be related to the performed experiment. For instance, example name of the OUT file storing the results returned by optimized NBC for fname dataset with 10000 of 2-dimensional points (records), where minPts = 5, Eps = 10, reference point r is constructed from minimal values in domains of all dimensions could be as follows:

    OUT_Opt-NBC_fname_D2_R10000_m5_e10_rMin.csv