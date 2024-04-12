#this script will remove all slurm.out files, really nice for keeping things clean! 
find . -type f -name "slurm-*.out" -delete
