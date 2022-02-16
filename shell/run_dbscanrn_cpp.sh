clear;
echo "Compilation DBSCANRN ...";
g++ -I /root/anaconda3/include/ -I /root/anaconda3/include/ src/dbscanrn.cpp -o bin/dbscanrn;
echo "Compilation complete";
./bin/dbscanrn
python src/calulate_STAT_OUT_DEBUG.py