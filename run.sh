export CUDA_VISIBLE_DEVICES=0
conda activate base

HERE=$(pwd)

STAGE_ROOT=/data/scratch-oc40/$USER/stage/dl_nf
NOW=$(date +%Y%m%d_%H%M%S)
RND_STR=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 6 ; echo '')
GIT_COMMIT=$(git rev-parse --short HEAD || echo 'no_git')
STAGE_NAME=${NOW}_${RND_STR}_${GIT_COMMIT}
STAGE_DIR=$STAGE_ROOT/$STAGE_NAME
mkdir -p $STAGE_DIR
rsync -av . $STAGE_DIR --exclude='.git' --exclude='*.pyc' --exclude=tmp
chmod 777 -R $STAGE_DIR
echo Staging done to $STAGE_DIR

LOG_DIR=$STAGE_DIR/logs
mkdir -p $LOG_DIR
echo Log dir: $LOG_DIR

echo Starting at $(date)
cd $STAGE_DIR

python -m wandb login 73f8ff40bb7f8589e9bd1f476196a896f662cdfa

python train.py --workdir=$LOG_DIR | tee -a $LOG_DIR/output.log
echo Finished at $(date)
echo check logs at $LOG_DIR/output.log

cd $HERE
