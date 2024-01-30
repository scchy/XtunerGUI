
pid=`ps -ef | grep -E "xtuner train.*" | grep -v "grep" | awk '{print $2}'`  
for id in $pid
do
    kill -9 $id
    echo "killed $id"
done;
