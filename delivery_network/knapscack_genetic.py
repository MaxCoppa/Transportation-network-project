from knapsack import *

"""
Genetic Algorithm Approach:
"""

def generate_individual(u, routes, trucks, budget):
    """
    Generate a random individual for the genetic algorithm.

    Parameters:
    - u: Union_set object, the union set representing the graph.
    - routes: list, a list containing information about routes.
    - trucks: dictionary, a dictionary containing information about trucks.
    - budget: int, the budget constraint.

    Returns:
    - list, a randomly generated individual.
    """
    individual = []
    for route in routes:
        start, end = route[0], route[1]
        truck = truck_assigned(trucks, u, start, end)
        if truck is not None and truck[1] <= budget:
            individual.append([start, end, truck[0]])
            budget -= truck[1]
    return individual  # Return the randomly generated individual

def generate_population(u, routes, trucks, budget, size):
    """
    Generate a random population of individuals for the genetic algorithm.

    Parameters:
    - u: Union_set object, the union set representing the graph.
    - routes: list, a list containing information about routes.
    - trucks: dictionary, a dictionary containing information about trucks.
    - budget: int, the budget constraint.
    - size: int, the size of the population.

    Returns:
    - list, a randomly generated population.
    """

    population = []
    for _ in range(size):
        individual = generate_individual(u, routes, trucks, budget)
        population.append(individual)
    return population  # Return the randomly generated population

def performance(individuals):
    """
    Calculate the fitness of each individual in the population.

    Parameters:
    - individuals: list, a list of individuals.

    Returns:
    - list, a list of fitness values for each individual.
    """
    fitness = []
    for individual in individuals:
        fit = sum([route[2] for route in individual])
        fitness.append(fit)
    return fitness  # Return the fitness values for each individual

def select_parents(individuals, perf):
    """
    Select two parents for reproduction using roulette wheel selection.

    Parameters:
    - individuals: list, a list of individuals.
    - perf: list, a list of fitness values for each individual.

    Returns:
    - tuple, containing the selected parents.
    """
    total_perf = sum(perf)
    r1 = random.uniform(0, total_perf)
    r2 = random.uniform(0, total_perf)
    i, j = 0, 0
    for k in range(len(perf)):
        r1 -= perf[k]
        if r1 <= 0:
            i = k
            break
    for k in range(len(perf)):
        r2 -= perf[k]
        if r2 <= 0:
            j = k
            break
    return individuals[i], individuals[j]  # Return the selected parents

def crossover(parent1, parent2):
    """
    Crossover two parents to create two children.

    Parameters:
    - parent1: list, the first parent.
    - parent2: list, the second parent.

    Returns:
    - tuple, containing the two children produced by crossover.
    """
    crossover_point = random.randint(0, len(parent1))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2  # Return the two children produced by crossover

def mutate(u, individual, routes, trucks, budget):
    """
    Mutate an individual by adding or removing a route.

    Parameters:
    - u: Union_set object, the union set representing the graph.
    - individual: list, the individual to be mutated.
    - routes: list, a list containing information about routes.
    - trucks: dictionary, a dictionary containing information about trucks.
    - budget: int, the budget constraint.

    Returns:
    - list, the mutated individual.
    """
    if random.random() < 0.5:
        remaining_budget = budget - sum([route[2] for route in individual])
        for route in routes:
            start, end = route[0], route[1]
            truck = truck_assigned(trucks, u, start, end)
            if truck is not None and truck[1] <= remaining_budget:
                individual.append((start, end, truck[0]))
                break
    else:
        if len(individual) > 0:
            index = random.randint(0, len(individual) - 1)
            removed_route = individual.pop(index)
            budget += removed_route[2]
    return individual  # Return the mutated individual

def evolve(u, population, perfs, routes, trucks, budget, mutation_rate):
    """
    Evolve the population by selecting parents, performing crossover, and mutation.

    Parameters:
    - u: Union_set object, the union set representing the graph.
    - population: list, a list of individuals representing the population.
    - perfs: list, a list of fitness values for each individual.
    - routes: list, a list containing information about routes.
    - trucks: dictionary, a dictionary containing information about trucks.
    - budget: int, the budget constraint.
    - mutation_rate: float, the probability of mutation.

    Returns:
    - list, the evolved population.
    """
    new_population = copy.deepcopy(population)
    for _ in range(len(population)):
        parent1, parent2 = select_parents(population, perfs)
        child1, child2 = crossover(parent1, parent2)
        if random.random() < mutation_rate:
            new_population[_] = mutate(u, child1, routes, trucks, budget)
        else:
            new_population[_] = child1
        _ += 1
        if _ < len(population):
            if random.random() < mutation_rate:
                new_population[_] = mutate(u, child2, routes, trucks, budget)
            else:
                new_population[_] = child2
            _ += 1
    return new_population  # Return the evolved population

def knapsack_genetic(g, routes, trucks, budget, size=100, mutation_rate=0.1, time_limit=60):
    """
    Implement the genetic algorithm to solve the knapsack problem.

    Parameters:
    - g: Graph object, the graph representing the network.
    - routes: list, a list containing information about routes.
    - trucks: dictionary, a dictionary containing information about trucks.
    - budget: int, the budget constraint.
    - size: int, the size of the population (default is 100).
    - mutation_rate: float, the probability of mutation (default is 0.1).
    - time_limit: int, the time limit for the algorithm (default is 60 seconds).

    Returns:
    - tuple, containing the selected individual and its performance.
    """
    _, u = kruskal(g)  # Get the minimum spanning tree and its union set
    population = generate_population(u, routes, trucks, budget, size)
    t0 = time.time()
    tf = time.time()
    max_performance = 0
    selected_individual = []
    while tf - t0 < time_limit:
        perfs = performance(population)
        population = evolve(u, population, perfs, routes, trucks, budget, mutation_rate)
        tf = time.time()
    for _ in range(len(population)):
        performance_ = sum([route[2] for route in population[_]])
        if performance_ > max_performance:
            max_performance = performance_
            selected_individual = population[_]
    return selected_individual, max_performance  # Return the selected individual and its performance
