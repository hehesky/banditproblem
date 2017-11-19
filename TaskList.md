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
### Bandit Setting (X.Sun)
#### Naive Method
- [x] Greedy (A/B testing) (comparison basis)
- [x] Epsilon Greedy
- [ ] Regret Epsilon Greedy: add optimistic initialisation
- [ ] [Epsilon Decreasing](https://algobeans.com/2016/01/14/how-to-gamble-effectively/)
        (Main idea: first 100 times random, then for 100+n-th time, epsilon = 1/n)
- [ ] Adaptive epsilon-greedy strategy based on value differences (VDBE) (replace epsilon decreasing, based on learning process.)(2010)
#### Other Methods
- [ ] UCB
- [ ] Probability matching (Thompson Sampling)
- [ ] Information state search
### MDP Setting
- [ ] Contextual-Epsilon-greedy strategy (2012)
- [ ] UCB
- [ ] Probability matching (Thompson Sampling)
- [ ] Information state search
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


