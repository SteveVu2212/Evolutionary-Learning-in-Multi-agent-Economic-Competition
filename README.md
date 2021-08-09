# Evolutionary Learning in Multi-agent Economic Competition

## Authors

Genetic algorithm learning and the cobweb model, Jasmina Arifovic (1994)

Estimating Auction Equilibria using Idividual Evolutionary Learning, Kevin James (2019)

Steve Vu

## 1. Introduction

The project applies the multi-agent genetic algorithm to model learning of economic agents in the Cobweb model, Jasmina Arifovic (1994). In the model, competitive firms update their decision rules about next-period production and sales before they observe a market price. Consumers base their decisions on the current market price. Thus, total quantity supplied and the exogenously given demand determine the price that clears the market.

In a competitive market, an economic agent have mutually competing ideas/strategies about what his behavior in a given environment should be. In each time period, he selects one strategy based on its performance against the others under predefined conditions. After implementing the idea and obtaining the market information, he is able to evaluate all of his alternative ideas via calculating forgone utilities. The agent uses this information in the process of updating his beliefs to assign higher probability of reproduction to ideas/strategies that yields higher values in the performance evaluation.


## 2. The Cobweb model

The Cobweb model represents a competitive market where there are n firms producing the same goods, employing the same technology, and facing the same cost function:

![](https://github.com/SteveVu2212/Evolutionary-Learning-in-Multi-agent-Economic-Competition/blob/main/images/cost%20calculation.png)

$C_{i,t} = xq_{i,t} + \frac{1}{2}ynq^2_{i,t}$

where $C_{i,t}$ is firm i's cost of a production for sale at time t, and $q_{i,t}$ is the quantity it produces for sale at time t.

The expected profit of an agent, $\Pi^e_{i,t}$, is:

$\Pi^e_{i,t} = P^e_{t}q_{i,t} - xq_{i,t} - \frac{1}{2}ynq^2_{i,t}$

where $P^e_{t}$ is the expected price of the good at t. Since at time (t-1), the price of the goods at time t, $P_{t}$ is not available, the decision about optimal $q_{i,t}$ must be based on the expectation of $P_{t}$, $P^e_{i,t}$.

The price $P_{t}$ that clears the market at time t is determined by the demand curve:

$P_{t} = A - B\sum_{i=1}^n q_{i,t}$

where A and B are exogenously given parameters of the model.

At time period t, a firm i makes a decision about its production using a binary string of finite length l, written over {0,1}.

![](https://github.com/SteveVu2212/Evolutionary-Learning-in-Multi-agent-Economic-Competition/blob/main/images/binary%20string.png)

A decoded and normalized value of a binary string i gives the value of the quantity produced by a firm i at time period t. For a string i of length l the decoding works in the following way:

$x_{i,t} = \sum_{k=1}^{l} a^k_{i,t}2^{k-1}$

where $a^k_{i,t}$ is the value (0,1) taken at the kth position in the string.

The quantity that firm i decides to produce and offer for sale at time period t:

$q_{i,t} = \frac{x_{i,t}}{K}$

where K is a coefficient chosen to normalize the value of $x_{i,t}$.

## 3. The multi-agent genetic algorithm (MAGA)

The multi-agent genetic algorithm is described in great detail in Kevin James (2019). Upon initialization, each agent selects a strategy at random to play for the first iteration. Once every agents has submitted a strategy, they can each evolve by using their forgone utility as a fitness function. The forgone utility is calculated by holding all other agents' chosen strategies constant, and replacing their own played strategy with the strategy in the population being evaluated.

![](https://github.com/SteveVu2212/Evolutionary-Learning-in-Multi-agent-Economic-Competition/blob/main/images/MAGAs.png)

Economic agents' decision rules are updated using four genetic operators, parent selection, crossover, mutation, and survival selection. Each iteration of the algorithm requires pairs of strategies to be selected. Those are chosen to be parents by combination with replacement.

After a pair of strategies have been determined, the new strategy creation algorithms, crossover and mutation, are run with probabilities, pcross and pmut, respectively. These works towards maintaining a degree of diversify in the population of strategies. In the project, I employ the two-point crossover with random cut points and the flip bit mutation to strategies represented by binary strings.

At the end, the application of the survival selection, tournament or best selection with probabilities, results in the reduction of rules' deviation from the quantity that maximizes profit at the market clearing price and in the reduction of the population variance over the course of a simulation.

The MAGA is computationally optimized by three additions, fitness caching, phenotype caching and count storage. While the fitness function is traditionally the most computationally expensive step, the MAGA automatically saves the fitness with each strategy, and reues the value.

Converting from a binary string to the quantity requires some computational time and can be cached with the strategy once it has been calculated. In addition, count storage is to exploit the duplication of strategies which end up being copies of one another. Rather than storing an array of the strategies, the unique strategies are paired with the frequency in the population.

## 4. Results

The MAGA results in the quick convergence of the algorithm to rational expectations equilibrium values. All strategies in a population become identical and the beliefs of all agents about how much to produce and offer for sale converge to the same value which is equal to the optimal quantities when the market price is known.

![](https://github.com/SteveVu2212/Evolutionary-Learning-in-Multi-agent-Economic-Competition/blob/main/images/price.png)
![](https://github.com/SteveVu2212/Evolutionary-Learning-in-Multi-agent-Economic-Competition/blob/main/images/quantity.png)