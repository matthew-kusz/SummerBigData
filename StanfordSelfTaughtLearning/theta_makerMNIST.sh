#PBS -N stlMNISTsize60000Lamb0.003Rho0.1Beta10
#PBS -l walltime=02:30:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=16GB
#PBS -j oe

# uncomment if using qsub
if [ -z "$PBS_O_WORKDIR" ] 
then
        echo "PBS_O_WORKDIR not defined"
else
        cd $PBS_O_WORKDIR
        echo $PBS_O_WORKDIR
fi
#
# Setup GPU code
module load python/2.7.8
#
# This is the command the runs the python script
python -u stlMNIST.py 30000 10 1000 >& outputStlRho0.1Beta10MNISTLambda0.003.log
