# Target values 
srun -w ault17 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'dqn_config.use_src_pred=True' 'set_target_steps=10000000' 'dqn_config.replay_mem_size=32768' 'variation="no recomputed target vals"' &
srun -w ault17 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'dqn_config.r_steps=1' 'dqn_config.use_src_pred=True' 'set_target_steps=10000000' 'dqn_config.replay_mem_size=32768' 'variation="no recomputed target vals and single step reward"' && fg
