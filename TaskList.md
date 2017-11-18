# Task List

### Primary Tasks
- [ ] Implement Every Algorithms
- [ ] Develope detailed test plans
- [ ] Analyze results
- [ ] Draw a conclusion
- [ ] Write Report/Paper

## Create Bandit
### Bandit (K.Zhou)
- [x] Bernoulli Bandit √
  - [x] Initial the parameters of Bernoulli distribution
  - [x] Pull and return a reward
- [x] Gaussian Bandit √
  - [x] Initial the parameters of Bernoulli distribution
  - [x] Pull and return a reward
- [x] Multiarmed Bandit
  - [x] Initial the type (Bernoulli, Gaussian) and arm numbers
  - [x] Pull one specific arm and return a reward

## Implement Algorithms
### Bandit strategies
#### Epsilon Algorithm (semi-uniform) (X.Sun)
- [ ] Epsilon First (comparison basis)
- [ ] Epsilon Greedy
- [ ] Epsilon Decreasing
- [ ] Adaptive epsilon-greedy strategy based on value differences (VDBE) (replace epsilon decreasing, based on learning process.)(2010)
- [ ] Contextual-Epsilon-greedy strategy (2012)
#### Probability matching strategies (Bayesian sampling)
#### Pricing strategies
#### Strategies with ethical constraints

### Contextual bandit
#### Algotithms
##### Online linear classifier
- [ ] UCB
##### Online non-linear classifier
- [ ] UCBogram algorithm
- [ ] NeuralBandit algorithm
- [ ] KernelUCB algorithm
- [ ] Bandit Forest algorithm:
#### Works
- [ ] Implementation
- [ ] Unit Test


## Current Test Plan
* Initial a ten armed Bernoulli Bandit with same distribution (will test on bernouli and other model)
* print the parameters of these bandit
* Rum times [10:100K] in logspace and draw lines (stability benefits...)

## Paper comparison
* Regression analysis
* Benefit analysis
* computing complexity analysis (time and space)


