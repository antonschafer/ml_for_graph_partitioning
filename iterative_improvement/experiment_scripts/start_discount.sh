# discount factor
srun -w ault18 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'algo_config.discount_factor=0' 'variation="discount 0"'  &
srun -w ault18 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'algo_config.discount_factor=0' 'dqn_config.r_steps=1' 'variation="discount 0 r_steps 1"' &
srun -w ault18 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'algo_config.discount_factor=0.5' 'variation="discount 0.5"'  &
srun -w ault18 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'algo_config.discount_factor=0.7' 'variation="discount 0.7"' &
srun -w ault18 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'algo_config.discount_factor=0.99' 'variation="discount 0.99"' && fg
