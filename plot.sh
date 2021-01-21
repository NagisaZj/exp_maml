CUDA_VISBLE_DEVICES=3 python main.py --env-name 2DPointEnvCustom-v1 --fast-batch-size 20 --meta-batch-size 40 --output-folder hcv-3 --num-workers 16 --embed-size 32  --exp-lr 7e-4 --baseline-type nn --nonlinearity tanh --num-layers-pre 1 --hidden-size 64 --seed 0 --M-type returns --exp-eps 3 --gpu 0

CUDA_VISBLE_DEVICES=3 python main.py --env-name AntRandGoalEnv-v1 --fast-batch-size 20 --meta-batch-size 40 --output-folder hcv-1 --num-workers 16 --embed-size 32  --exp-lr 7e-4 --baseline-type nn --nonlinearity tanh --num-layers-pre 1 --hidden-size 64 --seed 1 --M-type returns --exp-eps 3

CUDA_VISBLE_DEVICES=8 python main.py --env-name HalfCheetahRandVelEnv-v1 --fast-batch-size 20 --meta-batch-size 40 --output-folder hcv-new-3 --num-workers 16 --embed-size 32  --exp-lr 7e-4 --baseline-type nn --nonlinearity tanh --num-layers-pre 1 --hidden-size 64 --seed 3 --M-type returns --exp-eps 1 --gpu 8

CUDA_VISBLE_DEVICES=5 python main.py --env-name Walker2DRandVelEnv-v1 --fast-batch-size 20 --meta-batch-size 40 --output-folder hcv-1 --num-workers 16 --embed-size 32  --exp-lr 7e-4 --baseline-type nn --nonlinearity tanh --num-layers-pre 1 --hidden-size 64 --seed 1 --M-type returns --exp-eps 1

CUDA_VISBLE_DEVICES=0 python main.py --env-name Walker2DRandParamsEnv-v1 --fast-batch-size 20 --meta-batch-size 40 --output-folder hcv-11 --num-workers 16 --embed-size 32  --exp-lr 7e-4 --baseline-type nn --nonlinearity tanh --num-layers-pre 1 --hidden-size 64 --seed 1 --M-type returns --exp-eps 1 --gpu 0

CUDA_VISBLE_DEVICES=9 python main.py --env-name reacher-goal-sparse-v1 --fast-batch-size 20 --meta-batch-size 40 --output-folder hcv-1 --num-workers 16 --embed-size 32  --exp-lr 7e-4 --baseline-type nn --nonlinearity tanh --num-layers-pre 1 --hidden-size 64 --seed 1 --M-type returns --exp-eps 3 --gpu 9

CUDA_VISBLE_DEVICES=1 python main.py --env-name HopperRandParamsEnv-v1 --fast-batch-size 20 --meta-batch-size 40 --output-folder hcv-33 --num-workers 16 --embed-size 32  --exp-lr 7e-4 --baseline-type nn --nonlinearity tanh --num-layers-pre 1 --hidden-size 64 --seed 3 --M-type returns --exp-eps 1 --gpu 1

CUDA_VISBLE_DEVICES=1 python main.py --env-name metaworld-5 --fast-batch-size 20 --meta-batch-size 40 --output-folder hcv-33 --num-workers 16 --embed-size 32  --exp-lr 7e-4 --baseline-type nn --nonlinearity tanh --num-layers-pre 1 --hidden-size 64 --seed 3 --M-type returns --exp-eps 3 --gpu 1
