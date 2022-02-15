clear;
echo "Compilation DBSCANRN optimized ...";
g++ -I /root/anaconda3/include/ -I /root/anaconda3/include/ src/dbscanrn_optimized.cpp -o bin/dbscanrn_opt;
echo "Compilation complete";
./bin/dbscanrn_opt