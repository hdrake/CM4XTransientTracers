#!/bin/bash
#SBATCH --partition=compute
#SBATCH --mail-type=END
#SBATCH --mail-user=anthony.meza@whoi.edu
#SBATCH --ntasks=1
#SBATCH --mem=40gb
#SBATCH --time=02:00:00
#SBATCH --job-name jupyter_ameza
#SBATCH --output=log-jupyter-%j.log
# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')
# print tunneling instructions jupyter-log 
echo -e "
For more info and how to connect from windows,
see https://docs.ycrc.yale.edu/clusters-at-yale/guides/jupyter/
MacOS or linux terminal command to create your ssh tunnel
ssh -N -f -L ${port}:${node}:${port} ${user}@poseidon.whoi.edu
Windows MobaXterm info
Forwarded port:same as remote port Remote server: ${node}
Remote port: ${port}
SSH server: poseidon.whoi.edu
SSHlogin: $user
SSH port: 22
Use a Browser on your local machine to go to: localhost:${port} (prefix w/ https:// if using password) "
# load modules or conda environments here
# uncomment the two lines below to use your conda environment called notebook_env.
# Alternatively, activate your desired environment in the command line before you run this # script.
#module load anacomam
source ~/.bash_profile
conda activate cm4x_analysis
# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW
jupyter-lab --no-browser --port=${port} --ip=${node}
