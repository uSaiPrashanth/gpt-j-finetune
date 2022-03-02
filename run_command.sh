echo "Starting Distributed Finetuning for Gpt-J"

mapfile -t hostnames < <( cat /job/hosts )
rank=0

mastername=$(hostname -I)
echo "Master Address:" $mastername

for hostname in "${hostnames[@]}"
do
    echo $hostname
    echo $rank
    if (( $rank >= 1 )); then
        break
    fi
    output=-1
    while (( $output != $rank )); do
        output=$(rsh $hostname bash /home/mchorse/gpt-j-finetune/node_commands.sh $rank $mastername)
    done
    rank=$(($rank+1))
done