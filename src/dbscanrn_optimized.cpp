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
#include <xtensor/xsort.hpp>
#include <string>
#include <regex>

using namespace xt;
using namespace std;
using namespace std::chrono;
using json = nlohmann::json;

double current_time()
{
    auto current_time = std::chrono::system_clock::now();
    auto duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());
    double num_seconds = duration_in_seconds.count();
    return num_seconds;
}

float cosine_dissimilarity(const xarray<double> &a, const xarray<double> &b)
{
    auto numerator = sum(a * b);
    auto denominator = sqrt(sum(pow(a, 2))) * sqrt(sum(pow(b, 2)));
    auto result = numerator / denominator;
    return 1 - result(0);
}

float euclidean_distance(const xarray<double> &a, const xarray<double> &b)
{
    return (float)sqrt(sum(pow((a - b), 2))(0));
}

double epsilon_prim(double epsilon)
{
    return (double)sqrt(2 - 2 * epsilon);
}

auto normalize(const xarray<double> &a)
{
    xarray<double> result = a / sqrt(sum(pow(a, 2)));
    return result;
}

xarray<double> CLUSTER;
xarray<double> STATE;

int main()
{

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
        if (not conf["disable"].get<bool>())
        {
            std::cout << conf["name"].get<std::string>() << "\n";
            auto path = conf["path"].get<std::string>();
            auto k = conf["params_dbscanrn_opt"]["k"].get<int>();
            auto out_path = conf["out_path"].get<std::string>();
            auto log_out = conf["log_out"].get<std::string>();

            out_path = regex_replace(out_path, regex("algorithm"), "dbscanrn_opt_cpp");
            out_path = regex_replace(out_path, regex("similarity"), "euclidean_distance");
            log_out = regex_replace(log_out, regex("algorithm"), "dbscanrn_opt_cpp");
            log_out = regex_replace(log_out, regex("similarity"), "euclidean_distance");

            // load data
            ofstream outfile;
            outfile.open(log_out);

            auto start = chrono::high_resolution_clock::now();

            ifstream input_file;
            input_file.open(path);
            auto data = xt::load_csv<double>(input_file);
            input_file.close();

            auto end = chrono::high_resolution_clock::now();
            duration<double> elapsed = (end - start);
            outfile << std::setprecision(16) << current_time() << ",reading_data,"
                    << "," << elapsed.count() << "," << endl;

            data.reshape(data.shape());
            xarray<double> X = view(data, all(), range(0, -1));

            // init
            void clustering_algorithm(const xarray<float> &X_original, int k, std::ofstream &outfile);

            // run algorithm
            cout << "Processing ..."
                 << "\n";
            clustering_algorithm(X, k, outfile);
            start = chrono::high_resolution_clock::now();

            // save output
            ofstream out_file;
            out_file.open(out_path);
            dump_csv(out_file, xt::stack(xtuple(CLUSTER, STATE), 1));
            out_file.close();

            elapsed = (chrono::high_resolution_clock::now() - start);
            outfile << std::setprecision(16) << current_time() << ",writing_data,"
                    << "," << elapsed.count() << ',' << endl;
            outfile.close();
            std::cout << "Done."
                      << "\n";
        }
    }
}

void clustering_algorithm(const xarray<float> &X_original, int k, std::ofstream &outfile)
{
    outfile << std::setprecision(16) << current_time() << ",start log,,," << endl;

    // normalization
    auto start = chrono::high_resolution_clock::now();
    xarray<float> X = normalize(X_original);
    auto end = chrono::high_resolution_clock::now();
    duration<double> elapsed = (end - start);

    outfile << std::setprecision(16) << current_time() << ",normalization_time,"
            << "," << elapsed.count() << "," << endl;

    // each data point can be in one of 3 stages
    int NOT_VISITED = -1; // not visited point
    int VISITED = 0;      // non-core point
    int CLUSTERED = 1;    // core point

    // initial setup
    int n = X.shape()[0];
    xarray<double> cluster = ones<double>({n}) * -1;        // cluster register
    xarray<double> state = ones<double>({n}) * NOT_VISITED; // state register
    int cluster_id = 1;
    xarray<int> all_point_indices = arange(n);
    map<int, vector<int>> point_rnn;
    map<int, xarray<int>> point_knn;

    xarray<int> all_point_sorted;
    xarray<int> candidates;
    float real_max;
    xarray<double> pessimistic_estimation;
    xarray<double> candidate_distance;
    int down_row;

    function<void(int)> calc_pessimistic_estimation = [&](
                                                          int current_index) -> void
    {
        // choosing the next point to check if he is a better neighbor
        int new_candidate = all_point_sorted(down_row);
        bool previous_check = pessimistic_estimation(down_row) < real_max;
        down_row++;
        if (!previous_check)
        {
            point_knn[(int)current_index] = candidates;
            return;
        };

        if (previous_check)
        {
            float similarity = euclidean_distance(
                new_candidate, all_point_sorted[current_index]);
            outfile << std::setprecision(16) << current_time() << ",similarity_calculation," << (int)current_index << ",1," << endl;
            if (similarity < real_max)
            {
                candidates = filter(candidates, not_equal(candidate_distance, real_max));
                candidates = concatenate(xtuple(candidates, xarray<int>{new_candidate}));
                candidate_distance = concatenate(xtuple(candidate_distance, xarray<double>{similarity}));
                real_max = epsilon_prim(amax(candidate_distance)(0));
                calc_pessimistic_estimation((int)current_index);
            };
        };
        calc_pessimistic_estimation((int)current_index);
    };

    auto get_tiknn = [&](int current_index, xarray<int> neighbor_indices, int k)
    {
        auto start = chrono::high_resolution_clock::now();
        vector<double> r_dist;
        for (auto &idx : all_point_indices)
        {
            r_dist.push_back(euclidean_distance(row(X, idx), {0, 1}));
        };
        auto end = chrono::high_resolution_clock::now();
        duration<double> elapsed = end - start;

        outfile << std::setprecision(16) << current_time() << ",pessimistic_estimation_time," << current_index << "," << elapsed.count() << "," << endl;

        xarray<double> r_distance = adapt(r_dist, {r_dist.size()});
        auto current_index_r_distance = r_distance[current_index];

        start = chrono::high_resolution_clock::now();
        pessimistic_estimation = abs(current_index_r_distance - r_distance);
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;

        outfile << std::setprecision(16) << current_time() << ",dist_to_ref_point_time," << current_index << "," << elapsed.count() << "," << endl;

        start = chrono::high_resolution_clock::now();
        xarray<int> sort_indices = argsort(pessimistic_estimation);
        all_point_sorted = index_view(all_point_indices, sort_indices); // df
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;

        outfile << std::setprecision(16) << current_time() << ",sorting_pessimistic_est_time," << current_index << "," << elapsed.count() << "," << endl;

        // selecting candidates for k - nearest neighbors
        candidates = view(all_point_sorted, range(1, k + 1)); // dfx
        down_row = k + 2;

        // calculation of similarity for candidates
        vector<double> dist;
        for (auto &candidate_idx : candidates)
        {
            outfile << std::setprecision(16) << current_time() << ",similarity_calculation," << (int)current_index << ",1," << endl;
            dist.push_back(euclidean_distance(row(X, current_index), row(X, candidate_idx)));
        }
        candidate_distance = adapt(dist, {dist.size()});
        real_max = epsilon_prim(amax(candidate_distance)(0));

        calc_pessimistic_estimation((int)current_index);
    };

    auto get_knn = [&](int current_index, xarray<int> neighbor_indices, int k)
    {
        vector<float> neighbor_sim;
        for (auto &neighbor_index : neighbor_indices)
        {

            // outfile << std::setprecision (16) << current_time() << ",similarity_calculation," << (int) current_index << ",1," << endl;

            neighbor_sim.push_back(euclidean_distance(row(X, neighbor_index), row(X, current_index)));
        };
        xarray<float> neighbor_similaritys = adapt(neighbor_sim, {neighbor_sim.size()});
        xarray<int> sort_indices = argsort(neighbor_similaritys);
        xarray<int> neighbor_indices_sorted = index_view(neighbor_indices, sort_indices);
        point_knn[(int)current_index] = view(neighbor_indices_sorted, range(0, k));
    };

    auto get_pointwise_rnn = [&](int current_index)
    {
        vector<int> rnn;
        for (auto &neighbor_index : point_knn[(int)current_index])
        {
            if ((point_knn[(int)neighbor_index], (int)current_index))
            {
                rnn.push_back((int)neighbor_index);
            };
        };
        point_rnn[(int)current_index] = rnn;
    };

    auto get_rnn = [&]()
    {
        auto start = chrono::high_resolution_clock::now();
        for (int current_index = 0; current_index < n; current_index++)
        {
            // remove current index from neighbors
            xarray<int> neighbor_indices = col(
                from_indices(argwhere(not_equal(all_point_indices, current_index))), 0);
            get_tiknn((int)current_index, neighbor_indices, k);

            // printf("%f,knn_neighbors_id,%d,,",current_time(), current_index);
            //             outfile << std::setprecision (16) << current_time() << ",knn_neighbors_id," <<
            //                     current_index << ",,";
            //             for(auto& i : point_knn[(int) current_index]) {outfile << i << ";"; }
            //             outfile << endl;

            //             outfile << std::setprecision (16) << current_time() << ",|knn_neighbors|," <<
            //                     current_index << "," << (int) point_knn[(int) current_index].shape(0) << "," << endl;
        };
        auto end = chrono::high_resolution_clock::now();
        duration<double> elapsed = end - start;

        outfile << std::setprecision(16) << current_time() << ",knn_time,"
                << "," << elapsed.count() << ',' << endl;

        start = chrono::high_resolution_clock::now();
        for (int current_index = 0; current_index < n; current_index++)
        {
            get_pointwise_rnn((int)current_index);

            //             outfile << std::setprecision (16) << current_time() << ",rnn_neighbors_id," <<
            //                     current_index << ",,";
            //             for(auto& i : point_knn[(int) current_index]) {outfile << i << ";"; }
            //             outfile << endl;

            //             outfile << std::setprecision (16) << current_time() << ",|rnn_neighbors|," <<
            //                     current_index << "," << (int) point_knn[(int) current_index].shape(0) << "," << endl;
        };
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;
        outfile << std::setprecision(16) << current_time() << ",rnn_time,"
                << "," << elapsed.count() << ',' << endl;
    };

    // extend cluster
    function<void(int)> search = [&](int current_index) -> void
    {
        if (point_rnn[(int)current_index].size() < k)
        {
            state[(int)current_index] = VISITED;
        }
        else
        {
            state[(int)current_index] = CLUSTERED;
            cluster[(int)current_index] = cluster_id;
            for (auto &neighbor_index : point_rnn[(int)current_index])
            {
                if (state[(int)neighbor_index] == NOT_VISITED)
                {
                    search(neighbor_index);
                };
                state[(int)neighbor_index] = CLUSTERED;
                cluster[(int)neighbor_index] = cluster_id;
            };
        };
    };

    get_rnn();

    // visit all points
    while (any(equal(state, NOT_VISITED)))
    {
        xarray<int> current_index = from_indices(argwhere(equal(state, NOT_VISITED)));
        search(current_index(0, 0));
        cluster_id++;
        outfile << std::setprecision(16) << current_time() << ",knn_neighbors_id," << current_index(0, 0) << ",,";
        for (auto &i : point_knn[(int)current_index(0, 0)])
        {
            outfile << i << ";";
        }
        outfile << endl;
        outfile << std::setprecision(16) << current_time() << ",|knn_neighbors|," << current_index(0, 0) << "," << (int)point_knn[(int)current_index(0, 0)].shape(0) << "," << endl;

        outfile << std::setprecision(16) << current_time() << ",rnn_neighbors_id," << current_index(0, 0) << ",,";
        for (auto &i : point_knn[(int)current_index(0, 0)])
        {
            outfile << i << ";";
        }
        outfile << endl;

        outfile << std::setprecision(16) << current_time() << ",|rnn_neighbors|," << current_index(0, 0) << "," << (int)point_knn[(int)current_index(0, 0)].shape(0) << "," << endl;
    }

    // clusterize all outlier points to nearest cluster
    while (any(equal(state, VISITED)))
    {
        xarray<int> idx_not_clustered = from_indices(argwhere(equal(state, VISITED)));
        xarray<int> idx_clustered = from_indices(argwhere(equal(state, CLUSTERED)));
        get_knn((int)idx_not_clustered(0, 0), idx_clustered, 1);
        auto closest_clustered_idx = point_knn[(int)idx_not_clustered(0, 0)](0);
        cluster[(int)idx_not_clustered(0, 0)] = cluster[(int)closest_clustered_idx];
        state[(int)idx_not_clustered(0, 0)] = CLUSTERED;
    }
    outfile << std::setprecision(16) << current_time() << ",stop log,,," << endl;
    CLUSTER = cluster;
    STATE = state;
    // return xtuple(cluster, );
}