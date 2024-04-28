export PYTHONPATH=".":$PYTHONPATH
wandb_online="False"
entity=""

if [ ${wandb_online} == "True" ]; then
    export WANDB_API_KEY=""
    export WANDB_MODE="online"
else
    wandb offline
fi
