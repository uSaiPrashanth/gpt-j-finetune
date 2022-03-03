echo "Starting Distributed Finetuning for Gpt-J"
echo "Enter your Wandb login id"
read wandb_key
mapfile -t hostnames < <( cat /job/hosts )
rank=0
nnodes=2
mastername=$(hostname -I)
echo "Master Address:" $mastername

for hostname in "${hostnames[@]}"
do
    echo $hostname
    echo $rank
    if (( $rank >= $nnodes )); then
        break
    fi
    output=-1
    while (( $output != $rank )); do
        output=$(ssh $hostname bash /home/mchorse/gpt-j-finetune/node_commands.sh $nnodes $rank $mastername $wandb_key)
    done
    rank=$(($rank+1))
done