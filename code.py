import random
import operator
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

random.seed(42)

class Cobweb():
    def __init__(self, meta_parameters):
      self.x_parameter = meta_parameters.get("x_parameter")
      self.y_parameter = meta_parameters.get("y_parameter")
      self.A_parameter = meta_parameters.get("A_parameter")
      self.B_parameter = meta_parameters.get("B_parameter")
      self.num_firms = meta_parameters.get("num_firms")
    
    #Return a decoded value of a strategy
    def decoded_value(self, strategy):
      dvalue = 0
      for k in range(len(strategy)):
          dvalue += strategy[k] * pow(2, k)
      return dvalue

    #Return a quantity value of a strategy
    def quantity(self, dvalue, K_coefficient):
      return dvalue/K_coefficient

    #Return market price given quantities of submitted strategies of firms
    def market_clear_price(self, list_quantity):
      return self.A_parameter - self.B_parameter * sum(list_quantity)

    #Return a cost of a strategy
    def production_cost(self, quantity):
      return self.x_parameter * quantity + 0.5 * self.y_parameter * self.num_firms * pow(quantity, 2)
    
    #Return a profit of a strategy
    def profit(self, quantity, market_price, strategy):
      return market_price[strategy] * quantity - self.production_cost(quantity)

class Genetic_algorithm():
    def __init__(self, meta_parameters):
      self.pmut = meta_parameters.get("pmut")
      self.pcross = meta_parameters.get("pcross")
      self.num_strategies = meta_parameters.get("num_strategies")
      self.num_tournament = meta_parameters.get('num_tournament')

    #Generate pairs of strategies
    def parent_selection(self, list_strategies):
      return list(combinations_with_replacement(list_strategies, 2))

    #Mutate new strategies
    def mutation(self, strategy):
      s = list(strategy)
      flip_position = random.randrange(len(strategy))
      if random.random() < self.pmut:
        if s[flip_position] == 1:
          s[flip_position]= 0
        else:
          s[flip_position]= 1
      return tuple(s)
    
    #Generate new strategies
    #Each parent represents for a pair of two strategies
    def crossover(self, parents):
      if random.random() < self.pcross:
        div_position = sorted(random.sample(range(1, len(parents[0]) - 1), 2))
        new_strategy_1 = self.mutation(parents[1][:div_position[0]] + parents[0][div_position[0] : div_position[1]] + parents[1][div_position[1]:])
        new_strategy_2 = self.mutation(parents[0][:div_position[0]] + parents[1][div_position[0] : div_position[1]] + parents[0][div_position[1]:])
        return new_strategy_1, new_strategy_2
      else:
        return list()

    #Generate a new population of strategies
    def new_population(self, initial_strategies, firm):
        new_children = []
        parents = self.parent_selection(list(initial_strategies[firm].keys()))
        for i in parents:
            for j in self.crossover(i):
                new_children.append(j)

        #Adding new children to initial strategies to generate a new population of strategies
        new_strategies = initial_strategies[firm].copy()
        for strategy in new_children:
            if strategy in new_strategies:
                new_strategies[strategy] += 1
            else:
                new_strategies[strategy] = 1
        return new_strategies

    #Reproducing a population of strategies going into next round by comparing fitnesses/profit of strategies that are randomly choosen
    def tournament(self, all_profit):
        #Randomly choose #num_tournament of strategies
        strategies = np.array(list(all_profit.keys()))
        indices = np.random.choice(len(strategies), self.num_tournament)

        #Look up their profit from all_profit
        tournament_profit = {}
        for i in strategies[indices]:
          tournament_profit[tuple(i)] = all_profit[tuple(i)]
        
        #Sort the profit in decending order
        sorted_tournament_profit = dict(sorted(tournament_profit.items(), key=operator.itemgetter(1), reverse=True))

        #Return the highest profit strategy
        return list(sorted_tournament_profit.keys())[0]


    def strategy_selection(self, reproduced_strategy_profit, pselect):
        if random.random() < pselect:
          sorted_reproduced_profit = dict(sorted(reproduced_strategy_profit.items(), key=operator.itemgetter(1), reverse=True))
          return list(sorted_reproduced_profit.keys())[0]
        else:
          return self.tournament(reproduced_strategy_profit)

#Generating lists of competing strategies used to calculate forgone utilities given strategies of competitors
def comp_strategies(start_strategies, firm_strategies, firm):
  competing_strategies = {}
  for strategy in firm_strategies:
    competing_strategies[strategy] = ([strategy] + [x for i,x in enumerate(start_strategies) if i != firm])
  return competing_strategies

#Calculate quantities given list of competing strategies
def lst_quantity(firm_strategies, K_coefficient, competing_strategies, all_phenotype):
  list_quantity = {}
  for strategy in firm_strategies:
    list_quantity[strategy] = list(map(lambda x: x/K_coefficient, [all_phenotype[i] for i in competing_strategies[strategy]]))
  return list_quantity

#Calculate a market clear price given list of quantity
def lst_market_price(firm_strategies, list_quantity):
  market_price = {}
  for strategy in firm_strategies:
    market_price[strategy] = Model.market_clear_price(list_quantity[strategy])
  return market_price

#Calculate profit of population of strategies
def lst_profit(firm_strategies, market_price, list_quantity):
  all_profit = {}
  for strategy in firm_strategies:
    all_profit[strategy] = Model.profit(list_quantity[strategy][0], market_price, strategy)
  return all_profit

def initial_population(meta_parameters):
      #Generate random strategies
  raw_strategies = []
  for _ in range(meta_parameters['num_firms']):
      raw_strategies.append([tuple(random.randrange(2) for _ in range(meta_parameters['string_length'])) for _ in range(meta_parameters['num_strategies'])])

  #Structure those strategies to include the duplicated ones
  initial_strategies = []
  for firm in raw_strategies:
      dic = {}
      for strategy in firm:
          if strategy in dic:
              dic[strategy] += 1
          else:
              dic[strategy] = 1
      initial_strategies.append(dic)

  #Each firm randomly chooses an initial strategy from its population of strategies
  start_strategies = []
  for firm in initial_strategies:
      start_strategies.append(random.choice((*firm,)))
  
  return initial_strategies, start_strategies

#new_strategies include initial strategies and new born strategies
def nstrategies(initial_strategies, meta_parameters):
  new_strategies = []
  for firm in range(meta_parameters["num_firms"]):
      new_strategies.append(GA_agent.new_population(initial_strategies, firm))
  return new_strategies

#Calculate decoded value of all strategies
def all_pheno(meta_parameters, all_strategies):
    phenotypes = {}
    grouped_strategies = []
    for i in range(meta_parameters['num_firms']):
      grouped_strategies += list(all_strategies[i].keys())
    for strategy in grouped_strategies:
      phenotypes[strategy] = Model.decoded_value(strategy)
    return phenotypes

meta_parameters = {"num_firms": 3, "num_strategies": 20, "string_length": 7, "num_tournament": 3, 'num_iterations': 100, 'period_threshold': 50,
                    "A_parameter": 100, "B_parameter": 0.02, "x_parameter": 3, "y_parameter": 1, "pcross": 0.9, "pmut": 0.033}

GA_agent = Genetic_algorithm(meta_parameters)
Model = Cobweb(meta_parameters)

#Set up the normalizing coefficient (K)
max_quantity = meta_parameters['A_parameter']/(meta_parameters['B_parameter'] * meta_parameters['num_firms'])
full_capacity = [1 for _ in range(meta_parameters['string_length'])]
K_coefficient = Model.decoded_value(full_capacity) / max_quantity

initial_strategies, start_strategies = initial_population(meta_parameters)

num_iterations = meta_parameters['num_iterations']

evolution_profit = []
evolution_price = []
evolution_quantity = []

for period in range(num_iterations):

  new_strategies = nstrategies(initial_strategies, meta_parameters)
  all_phenotype = all_pheno(meta_parameters, new_strategies)

  new_initial_strategies = []
  new_start_strategies = []

  profit = []
  price = []
  quantity = []

  for firm in range(meta_parameters["num_firms"]):
    
    firm_strategies = list(new_strategies[firm].keys())

    competing_strategies = comp_strategies(start_strategies, firm_strategies, firm)
    list_quantity = lst_quantity(firm_strategies, K_coefficient, competing_strategies, all_phenotype)
    market_price = lst_market_price(firm_strategies, list_quantity)
    all_profit = lst_profit(firm_strategies, market_price, list_quantity)
    
    survival = {}
    for _ in range(meta_parameters['num_strategies']):
      reproduced_strategy = GA_agent.tournament(all_profit)
      survival[reproduced_strategy] = new_strategies[firm][reproduced_strategy]

    reproduced_strategy_profit = {}
    for j in list(survival.keys()):
      reproduced_strategy_profit[j] = all_profit[j]

    #Use tournament to choose a strategy to play next round
    if period < meta_parameters['period_threshold']:
      new_start_strategies.append(GA_agent.strategy_selection(reproduced_strategy_profit, 0.5))
    else:
      new_start_strategies.append(GA_agent.strategy_selection(reproduced_strategy_profit, 0.8))

    #Survived strategies go to next round
    new_initial_strategies.append(survival)

    profit.append(int(all_profit[start_strategies[firm]]))
    quantity.append(list_quantity[start_strategies[firm]])
    price.append(int(market_price[start_strategies[firm]]))
  
  initial_strategies = new_initial_strategies
  start_strategies = new_start_strategies

  evolution_profit.append(profit)
  evolution_price.append(price)
  evolution_quantity.append(quantity)


firm_price = []
for i in range(meta_parameters['num_firms']):
  price = []
  for j in range(len(evolution_price)):
    price.append(evolution_price[j][i])
  firm_price.append(price)

firm_quantity = []
for i in range(meta_parameters['num_firms']):
  q = []
  for j in range(len(evolution_quantity)):
    q.append(evolution_quantity[j][i][0])
  firm_quantity.append(q)

period = range(25)
plt.figure(figsize= (8,6), dpi = 100)
plt.plot(period, firm_quantity[0][:25], label = 'firm 1', marker = 'o')
plt.plot(period, firm_quantity[1][:25], label = 'firm 2', marker = '+')
plt.plot(period, firm_quantity[2][:25], label = 'firm 3', marker = '', linestyle = 'dashed')
plt.xlabel("period")
plt.ylabel("quantity")
plt.legend(loc="upper right")
plt.show()

fig = plt.figure(figsize=(8,6), dpi=100)
ax = fig.add_subplot()
fig.subplots_adjust(top=0.7)
ax.annotate('rational expectation price', xy=(55, 95), xytext=(30, 80),
            arrowprops=dict(facecolor='black', shrink=0.005))
ax.set_xlabel('period')
ax.set_ylabel('market price')
ax.plot(firm_price[0][0:60])
plt.show()