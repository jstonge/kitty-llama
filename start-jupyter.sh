#!/bin/bash
joboutname=$1
out_dir=tmp
mkdir -p $out_dir
jup_outfile=$out_dir/$joboutname.out
>$jup_outfile  # clear the output file

sbatch --output=$jup_outfile jupyter-server.sbatch  # start the jupyter server on the cluster

echo "Jupyter notebook server starting on compute node, waiting..."
# wait for notebook server to start
while ! grep -q "http://localhost" $jup_outfile; do
    sleep 1
done

# get port
port=$(grep "http://localhost" $jup_outfile | sed -E 's/.*localhost:([0-9]+).*/\1/' | head -1)
# get token
token=$(grep -oP 'token=\K[^ ]+' $jup_outfile | head -1)
# get user
user=$(head -n 1 $jup_outfile | sed 's/.*\[\(.*\)\].*/\1/' | head -1)
# get host
host=$(head -n 2 $jup_outfile | tail -n 1 | sed 's/.*\[\(.*\)\].*/\1/' | head -1)

echo "Jupyter notebook server started on compute node $user@$host"
sleep 1

# connect to notebook server
ssh -fNT -L $port:localhost:$port $user@$host & 

echo "Port forwarding started $user@$host:$port -> localhost:$port"
echo ""
echo "connect to notebook server at:"
echo "  http://localhost:$port/?token=$token"
echo ""
