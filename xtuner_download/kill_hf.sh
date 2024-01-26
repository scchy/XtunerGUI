

rp_tp=$1;

pid=`ps -ef | grep -E "huggingface-cli.*?${rp_tp}" | grep -v "grep" | awk '{print $2}'`  
for id in $pid
do
    kill -9 $id
    echo "killed $id"
done;
