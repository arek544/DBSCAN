echo "Compilation DBSCAN ...";
g++ -I /root/anaconda3/include/ -I /root/anaconda3/include/ src/dbscan.cpp -o bin/dbscan;
echo "Compilation complete";
./bin/dbscan

echo "Compilation DBSCANRN ...";
g++ -I /root/anaconda3/include/ -I /root/anaconda3/include/ src/dbscanrn.cpp -o bin/dbscanrn;
echo "Compilation complete";
./bin/dbscanrn

echo "Compilation DBSCANRN optimized ...";
g++ -I /root/anaconda3/include/ -I /root/anaconda3/include/ src/dbscanrn_optimized.cpp -o bin/dbscanrn_opt;
echo "Compilation complete";
./bin/dbscanrn_opt

python src/calulate_STAT_OUT_DEBUG.py