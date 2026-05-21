conda activate rl_env

tensorboard --logdir /Users/luciobaiocchi/projects/FAIML-RL-26_g44/part2/tb_logs/


# SAC + HER
python train_sb3_sac.py --sampling-strategy none --env-type source --timesteps 100000 --her


python train_sb3_sac.py \
--sampling-strategy none \
--env-type source \
--timesteps 100000 \
--her \
--gradient-steps 4 \
--batch-size 512 \
--learning-rate 1e-3 \
--device mps
