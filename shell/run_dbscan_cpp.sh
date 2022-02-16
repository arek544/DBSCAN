clear;
echo "Compilation DBSCAN ...";
g++ -I /root/anaconda3/include/ -I /root/anaconda3/include/ src/dbscan.cpp -o bin/dbscan;
echo "Compilation complete";
./bin/dbscan
python src/calulate_STAT_OUT_DEBUG.py