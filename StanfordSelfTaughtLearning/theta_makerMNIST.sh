#PBS -N stlMNISTsize60000Lamb3e-3Rho0.1Beta3
#PBS -l walltime=02:00:00
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
python -u stlMNIST.py $PBS_ARRAYID 10 300 >& output_logs/outputStlRho0.1Beta3MNISTLambda$PBS_ARRAYID.log
