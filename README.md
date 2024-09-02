# Efficient and Stable Offline-to-online Reinforcement Learning via Continual Policy Revitalization

The official code for [Efficient and Stable Offline-to-online Reinforcement Learning via Continual Policy Revitalization](https://www.ijcai.org/proceedings/2024/0477.pdf), (IJCAI'24).

## Install Dependency

The training dependencies could be installed by the following command with conda. Notice that since we provide the same package in our training, it might be possible that the installed version of CUDA is not compatible with your GPU. In that case, you can mannually reinstall pytorch only.

```bash
conda env create -f environment.yml
```

To install the [D4RL](https://github.com/Farama-Foundation/D4RL) benchmark, try the following command

```bash
git clone https://github.com/Farama-Foundation/D4RL.git
cd d4rl
pip install -e .
```

## Training

### Experiment Setup

If you do not want to use wandb for tracking, you can run the following command in your terminal

```bash
wandb offline
```

Otherwise, you can fill the wandb account setting in `scripts/config.sh`

```bash
export PYTHONPATH=".":$PYTHONPATH
wandb_online="False"
entity=""

if [ ${wandb_online} == "True" ]; then
    export WANDB_API_KEY=""
    export WANDB_MODE="online"
else
    wandb offline
fi
```

### Offline Training

Run the following script to finish the offline experiments

```bash
bash ./script/run_td3bc_offline.sh tasktask quality namename seed --device $device_id
```

Value for the arguments

* task: halfcheetah, hopper, walker2d, all
* quality: medium, medium-replay, medium-expert, random
* name: original(paper args), corl(CORL args, recommended)
* seed: random seed
* device_id: cuda device ID

One example command is

```bash
bash ./script/run_td3bc_offline.sh halfcheetah medium corl 0 --device "cuda:0"
```

### Online Training

**Notice**: Online training is only possible after the corresponding offline training checkpoint is produced.

Run the following script to reproduce online experiments

```bash
bash ./script/run_cpr_online.sh tasktask quality original seed−−deviceseed --device device_id
```

Value for the arguments

* task: halfcheetah, hopper, walker2d, all
* quality: medium, medium-replay, medium-expert, random
* seed: random seed
* device_id: cuda device ID

One example command is

```bash
bash ./script/run_cpr_online.sh halfcheetah medium original 0 --device "cuda:0"
```

## See Results

The logs and models are stored in "./out" folder.

```bash
tensorboard --logdir="./out"
```

## Credits

We thank the following repos for the help:

* [OfflineRL-Lib](https://github.com/typoverflow/OfflineRL-Lib) provides the framework and implementation of most baselines.
* [CORL](https://github.com/tinkoff-ai/CORL) provides finetuned hyper-parameters.

## Citation

If you find this work useful for your research, you can cite with the following bib:

```bib
@inproceedings{
    cpr,
    title={Efficient and Stable Offline-to-online Reinforcement Learning via Continual Policy Revitalization},
    author={Rui Kong, Chenyang Wu, Chen-Xiao Gao, Zongzhang Zhang and Ming Li},
    booktitle={Proceedings of the Thirty-Third International Joint Conference on
    Artificial Intelligence, {IJCAI} 2024},
    year={2024},
}
```
