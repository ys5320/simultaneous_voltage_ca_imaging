#PBS -lselect=1:ncpus=32:mem=64gb
#PBS -lwalltime=08:00:00
#PBS -N results_2
#PBS -J 1-24

module load anaconda3/personal
source activate cancer_HPC_tst

cd ${PBS_O_WORKDIR}/../../

## RDS comments: if the above gives an error for path, try the following alternative
## by uncommenting the following line and checking the jobscript output/error files for the "DBG:" line

## cd ${PBS_O_WORKDIR}/.. ; echo "DBG: working dir is now: ${PWD}"


python create_paper_data.py $PBS_ARRAY_INDEX 0