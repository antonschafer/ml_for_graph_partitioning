# use progress 
srun -w ault18 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'use_progress=False' 'dqn_config.r_steps=1' 'variation="no progress input"'

