#PBS -N stlMNISTsize60000Lamb0.03Rho0.1Beta0.5TESTING
#PBS -l walltime=00:30:00
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
python -u stlMNIST.py 300000 10 50 >& outputStlRho0.1Beta0.5MNISTLambda0.03TESTING.log
