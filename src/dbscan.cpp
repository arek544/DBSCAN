#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xchunked_array.hpp>
#include <xtensor/xfixed.hpp>
#include <cstddef>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <istream>
#include <fstream>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <iostream>
#include <fstream>
#include <functional>
#include <cstdio>
#include <xtensor/xcsv.hpp>
#include <iostream>
#include <chrono>
#include <unistd.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <chrono>
#include <list>
#include <string>
#include <regex>

    
using namespace xt;
using namespace std;
using namespace std::chrono;
using json = nlohmann::json;


double current_time(){
    auto current_time = std::chrono::system_clock::now();
    auto duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());
    double num_seconds = duration_in_seconds.count();
    return num_seconds;
}

float euclidean_distance(const xarray<double>& a, const xarray<double>& b){ 
    return (float) sqrt(sum(pow((a - b), 2))(0));
}

double epsilon_prim(double epsilon){ 
    return (double) sqrt(2 - 2 * epsilon);
}

auto normalize(const xarray<double>& a){ 
    xarray<double> result = a / sqrt(sum(pow(a, 2)));
    return result;
}

xarray<double> CLUSTER; 
xarray<double> STATE;
    

int main() {
    
    string path;
    double epsilon;
    int minPts;
    string out_path;
    string log_out;
    
    // read configs
    std::ifstream file("configs/dbscan.json");
    json config = json::parse(file);
    
    for (auto conf : config)
    {
        if (not conf["disable"].get<bool>()) {
            std::cout << conf["name"].get<std::string>() << "\n";
            auto path = conf["path"].get<std::string>();
            auto epsilon = conf["params_dbscan"]["epsilon"].get<double>();
            auto minPts = conf["params_dbscan"]["minPts"].get<int>();
            auto out_path = conf["out_path"].get<std::string>();
            auto log_out = conf["log_out"].get<std::string>();
            
            epsilon = epsilon_prim(epsilon);
                
            out_path = regex_replace(out_path, regex("algorithm"), "dbscan_cpp");
            log_out = regex_replace(log_out, regex("algorithm"), "dbscan_cpp");
            
            // load data
            ofstream outfile;
            outfile.open(log_out);

            auto start =  chrono::high_resolution_clock::now();

            ifstream input_file;
            input_file.open (path);
            auto data = xt::load_csv<double>(input_file);
            input_file.close();

            auto end =  chrono::high_resolution_clock::now();
            duration<double> elapsed = (end - start);
            outfile << std::setprecision (16) << current_time() << ",reading_data," <<
                "," << elapsed.count() << "," << endl;

            data.reshape(data.shape());
            xarray<double> X = view(data, all(), range(0, -1));

            // init
            void dbscan(const xarray<float>& X, float epsilon, int minPts, std::ofstream& outfile);

            // run algorithm
            std::cout << "Processing ..." << "\n";
            dbscan(X, epsilon, minPts, outfile);
            start = chrono::high_resolution_clock::now();

            // save output
            ofstream out_file;
            out_file.open(out_path);
            dump_csv(out_file, xt::stack(xtuple(CLUSTER, STATE), 1));
            out_file.close();

            elapsed = (chrono::high_resolution_clock::now() - start);
            outfile << std::setprecision (16) << current_time() << ",writing_data," <<
                "," << elapsed.count() << ',' << endl;
            outfile.close();
            std::cout << "Done." << "\n";
        }
    }
}


void dbscan(const xarray<float>& X_original, float epsilon, int minPts, std::ofstream& outfile)
{
    outfile << std::setprecision (16) << current_time() << ",start log,,," << endl;

    // normalization
    auto start =  chrono::high_resolution_clock::now();
    xarray<float> X = normalize(X_original);
    auto end =  chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    outfile << std::setprecision (16) << current_time() << ",normalization_time," <<
        "," << elapsed.count() << "," << endl;
    
    // each data point can be in one of 3 stages
    int NOT_VISITED = -1; // not visited point
    int VISITED = 0; // non-core point
    int CLUSTERED = 1; // core point

    // initial setup
    int n = X.shape()[0];
    xarray<double> cluster = ones<double>({n}) * -1; // cluster register
    xarray<double> state = ones<double>({n}) * NOT_VISITED; // state register
    int cluster_id = 1;
        
    auto get_neighbors = [&] (int current_index) {
        list<int> neighbor_indices = list<int>();
        for (int neighbor_index : arange(0, n)){
            
            outfile << std::setprecision (16) << current_time() << ",similarity_calculation," <<
                current_index << ",1," << endl;
            
            if (euclidean_distance(row(X, neighbor_index), row(X, current_index)) <= (float) epsilon){
                neighbor_indices.push_back((int) neighbor_index);
            };
        }
        return neighbor_indices;
    };
    
    // extend cluster
    function < void( int ) > search = [&] (int current_index) -> void {
        auto start =  chrono::high_resolution_clock::now();
        list<int> neighbor_indices = get_neighbors(current_index);
        auto end = chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        outfile << std::setprecision (16) << current_time() << ",Eps_time," <<
                current_index << "," << elapsed.count() << ',' << endl;
        outfile << std::setprecision (16) << current_time() << ",|Eps_neighbors|," <<
                current_index << "," << (int) neighbor_indices.size() << ',' << endl;
        outfile << std::setprecision (16) << current_time() << ",Eps_neighbor_id," <<
                current_index << ",,";
        for(auto& i : neighbor_indices) {outfile << i << ";"; }
        outfile << endl;

        if ((int) neighbor_indices.size() >= minPts){
            state(current_index) = CLUSTERED;
            cluster(current_index) = cluster_id;
            for (auto& neighbor_index : neighbor_indices) {
                if (state(neighbor_index) == VISITED || state(neighbor_index) == NOT_VISITED){
                    state(neighbor_index) = CLUSTERED;
                    cluster(neighbor_index) = cluster_id;
                    search(neighbor_index);
                };
            };
        } else {
            state(current_index) = VISITED;
        };
    };

    while (any(equal(state, NOT_VISITED))){
        xarray<int> idx = from_indices(argwhere(equal(state, NOT_VISITED)));
        search(idx(0,0));
        cluster_id++;
    }

    outfile << std::setprecision (16) << current_time() << ",stop log,,," << endl;
    CLUSTER = cluster;
    STATE = state;
}
