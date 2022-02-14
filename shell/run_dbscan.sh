clear;
echo "Compilation DBSCAN ...";
g++ -I /root/anaconda3/include/ -I /root/anaconda3/include/ src/dbscan.cpp -o bin/dbscan;
echo "Compilation complete";
./bin/dbscan