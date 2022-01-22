export NAME="dbscan"
docker build -t $NAME -f Dockerfile .
docker stop $NAME
docker rm $NAME
docker run -it -p 8888-9000:8888-9000 -v $PWD:/work --name=$NAME $NAME    