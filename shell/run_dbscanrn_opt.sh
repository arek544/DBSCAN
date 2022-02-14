clear;
echo "Compilation DBSCANRN optimized ...";
g++ -I /root/anaconda3/include/ -I /root/anaconda3/include/ src/dbscanrn_optimized.cpp -o src/dbscanrn_optimized;
echo "Compilation complete";
./src/dbscanrn_optimized