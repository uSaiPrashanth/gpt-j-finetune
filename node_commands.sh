nnodes=$1
node_rank=$2
master_addr=$3

echo $node_rank

output=$(screen -dm bash -c "cd /home/mchorse/gpt-j-finetune; python3 -m torch.distributed.launch --nnodes $nnodes --node_rank $node_rank --nproc_per_node=1 --master_addr=$master_addr --master_port=4224 --logdir ./logs --use_env distributed_train.py --nnodes $nnodes --rank $node_rank")
