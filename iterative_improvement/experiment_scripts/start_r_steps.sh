# reward steps
srun -w ault18 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'dqn_config.r_steps=1' 'variation="r_steps 1"' &
srun -w ault18 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'dqn_config.r_steps=5' 'variation="r_steps 5"'  &
srun -w ault18 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'dqn_config.r_steps=10' 'variation="r_steps 10"'  && fg
