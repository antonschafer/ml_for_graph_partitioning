# K
srun -w ault18 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'algo_config.k=3' 'variation="k 3"' &
srun -w ault18 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'algo_config.k=4' 'variation="k 4"' &
srun -w ault18 --partition=testing --time 24:00:00 python -m iterative_improvement.train with 'algo_config.k=8' 'variation="k 8"' && fg
