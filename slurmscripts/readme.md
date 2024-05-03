# Slurm note


## Slurm experiments

### Datasets

LLaVa instruction data image prefixes:
- coco
- textvqa
- vg
- gqa
- ocr_vqa

Due to the large size of other datasets, here we sample only from TextVQG:
- Download the TextVQA data from LLaVa project
- Randomly sample 10k from `textvqa/*` instances from `llava_v1_5_mix665k.json` -> `llava_v1_5_mix665k---textvqa-10k.json`
    - Randomly sample 5k from the above 10k -> `llava_v1_5_mix665k---textvqa-5k.json`
- Append VLGuard data into the above files
- *Remember to put images in ./playground/data


### Commands for jobs

Submit job
```bash
sbatch -p gpu slurmscripts/trial_run.sh
sbatch -p gpu slurmscripts/run.sh
```

> Slurm header

```bash
#!/bin/bash
#
#SBATCH --job-name=olive
#SBATCH --output=./slurmlogs/%x-%j.output
#SBATCH --mail-user=thy.tran@tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task==8
#SBATCH --mem=64GB

# Activate and install basic packages.
source /storage/ukp/work/$USER/slurm_cmds/load_env.sh
source /storage/ukp/work/$USER/slurm_cmds/load_conda.sh
```


### Interactive run

Interactive session:
```bash
srun -p gpu --cpus-per-task 4 --mem=100GB --gres=gpu:1 --constraint="gpu_model:a6000" --pty bash
```

```bash
source /storage/ukp/work/$USER/slurm_cmds/load_env.sh
source /storage/ukp/work/$USER/slurm_cmds/load_conda.sh
conda activate llava
export PYTHONPATH="${PYTHONPATH}:/storage/ukp/work/thytran/sweetolive/LLaVA_Attack"
```


```bash
python llava/eval/run_llava.py \
--model-path ./checkpoints/78076-llava-v1.5-13b-task-lora--textvqa-5k--vlguard \
--model-base liuhaotian/llava-v1.5-13b \
--image-file ./playground/data/hatefulMemes/01247.png \
--query “why was this photo taken?”
```

## Notes/Questions

What is `--version v1` in the script arguments?

