export CUDA_VISIBLE_DEVICES=0
conda activate base

source config.sh

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

export WANDB_CONFIG_DIR=$STAGE_DIR/.wandb_config      # 强制 wandb 使用可写目录
mkdir -p $WANDB_CONFIG_DIR
chmod 700 $WANDB_CONFIG_DIR
export WANDB_START_METHOD=thread                      # 推荐，减少多进程问题
export WANDB_MODE=online

/data/scratch-oc40/zhh24/anaconda3/bin/python -m wandb login $WANDB_API_KEY
sleep 1
/data/scratch-oc40/zhh24/anaconda3/bin/python -m wandb login

python -u train.py --workdir=$LOG_DIR 2>&1 | tee -a $LOG_DIR/output.log
echo Finished at $(date)
echo check logs at $LOG_DIR/output.log

cd $HERE
