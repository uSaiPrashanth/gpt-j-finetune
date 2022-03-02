nnodes=1
node_rank=$1
master_addr=$2

echo $node_rank
cd /home/mchorse/gpt-j-finetune

output=$(screen -dm bash -c 'python3 -m torch.distributed.launch --nnodes $nnodes --node_rank $node_rank --nproc_per_node=1 --master_addr=$master_addr --master_port=4224 --use_env demo.py --nnodes $nnodes --rank $node_rank')