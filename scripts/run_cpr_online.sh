# run_cpr_benchmark.sh
all_args=("$@")
task=$1
quality=$2
config=$3
seed=$4
rest_args=("${all_args[@]:4}")

source scripts/config.sh
project="cpr-$phase"

if [ $task == "all" ]; then
    tasks=( "halfcheetah" "hopper" "walker2d" )
else
    tasks=( $task )
fi

if [ $quality == "all" ]; then
    qualities=( "medium" "medium-replay" "medium-expert" "random" )
else
    qualities=( $quality )
fi

for t in ${tasks[@]}; do
    for q in ${qualities[@]}; do
        echo python3 reproduce/online/run_cpr_online.py --config reproduce/online/config/$config/mujoco/${t}-${q}-v2.py --seed $seed --wandb.entity $entity --wandb.project $project ${rest_args[@]}
        python3 reproduce/online/run_cpr_online.py --config reproduce/online/config/$config/mujoco/${t}-${q}-v2.py --seed $seed --wandb.entity $entity --wandb.project $project ${rest_args[@]}
    done
done
