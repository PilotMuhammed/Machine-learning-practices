import numpy as np

def fitness_function(solution):
    return np.sum(solution)

def create_population(population_size, solution_size):
    population = np.random.randint(2, size=(population_size, solution_size))
    return population

def selection(population, fitness_values):
    probabilities = fitness_values / np.sum(fitness_values)
    selected_indices = np.random.choice(range(len(population)), size=len(population) , p=probabilities)
    selected_population = population[selected_indices]
    return selected_population 


def crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        offspring.extend([child1, child2])
    return np.array(offspring)


def mutation(offspring, mutation_rate):
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            if np.random.rand() < mutation_rate:
                offspring[i][j] = 1 - offspring[i][j]
    return offspring


def genetic_algorithm(population_size, solution_size, generations, mutation_rate):
    population = create_population(population_size, solution_size) 
    for _ in range(generations):
        fitness_values = np.array([fitness_function(solution) for solution in population])
        parents = selection(population, fitness_values)
        offspring = crossover(parents)

        mutated_offspring = mutation(offspring, mutation_rate)

        population = mutated_offspring
    best_solutions = population[np.argmax(fitness_values)]
    return best_solutions



population_size = 100
solution_size = 10
generations = 100
mutation_rate = 0.01

# best_solution = genetic_algorithm(population_size, solution_size, mutation_rate)
best_solution = genetic_algorithm(population_size, solution_size, generations, mutation_rate)
print("Best solution", best_solution)
print("fitness valus", fitness_function(best_solution))
