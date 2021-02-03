# Bachelor Thesis: Machine Learning for Graph Partitioning

## Abstract

Balanced Graph Partitioning, i.e. the partitioning of a graph into k partitions of roughly equal size while minimizing the number of crossing edges, is a highly relevant combinatorial optimization problem due its applications in parallel processing and other fields. As the problem is NP-hard, it is usually solved approximately using specialized heuristic algorithms. These heuristics are hand-engineered to enable good solu- tions in short time for real-world inputs. For a variety of combinatorial optimization problems, machine learning methods have recently been applied succesfully to learn such heuristics.
In this work, we propose two approaches for using machine learning in balanced graph partitioning algorithms. We consider general fea- tureless graphs and propose a new model architecture for generating node embeddings which proves useful for both approaches. The first approach relies entirely on an end-to-end machine learning model that assigns each node to a partition in a classification-style setup. The second approach is based on a meta algorithm that iteratively improves a given partitioning by swapping nodes’ partition assignments. These nodes are selected according to a learned heuristic. To compute this heuristic, we use a neural network that is trained via Deep Q-Learning. We achieve an asymptotic speedup by splitting the Q-function in two parts.
Both approaches outperform the random baseline by a large margin. The simpler end-to-end approach performs consistently worse than the algorithmic baselines, while the more algorithmically oriented iterative improvement approach produces better partitionings than a greedy al- gorithm and is close in performance to more specialized solvers. We see potential for significant improvements upon these results in future work builiding on our methods.

## Full Thesis

https://github.com/antonschafer/ml_for_graph_partitioning/raw/main/thesis.pdf

## Code Structure

- `graph_generation` contains scripts to generate the random graphs and run the baselines. 
- `greedy` contains C++ greedy baseline
- `embeddings` contains embedding generation methods and simpler end-to-end model
- `iterative_improvement` contains the Deep Q Learning method that is the main contribution
- the other files and directories contain other functionality like NN layers, dataset code, utility functions, and logging setup

All experiments were tracked with the `sacred` library and logged to a mongodb on a cloud server.

