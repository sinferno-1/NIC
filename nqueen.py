# # # NQueen: 

import random

class NQueensGeneticAlgorithm:
    def __init__(self, population_size, n, crossover_rate, mutation_rate):
        self.population_size = population_size
        self.n = n
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        return [random.sample(range(self.n), self.n) for _ in range(self.population_size)]

    def fitness(self, chromosome):
        attacks = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if chromosome[i] == chromosome[j] or abs(chromosome[i] - chromosome[j]) == abs(i - j):
                    attacks += 1
        return (self.n * (self.n - 1)) / 2 - attacks  # Maximum non-attacking pairs - number of attacking pairs

    def single_point_crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.n - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(self.n), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

    def select_parents(self, population):
        # Tournament selection
        tournament_size = 2
        selected_parents = []
        for _ in range(self.population_size):
            participants = random.sample(population, tournament_size)
            selected_parents.append(max(participants, key=self.fitness))
        return selected_parents

    def evolve(self, generations):
        population = self.initialize_population()
        for generation in range(generations):
            new_population = []
            parents = self.select_parents(population)
            for i in range(0, self.population_size, 2):  # Create new population
                parent1, parent2 = parents[i], parents[i + 1]
                if random.random() < self.crossover_rate:
                    child1, child2 = self.single_point_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                new_population.extend([child1, child2])
            # Mutation
            for individual in new_population:
                self.mutate(individual)
            population = new_population
        best_solution = max(population, key=self.fitness)
        return best_solution

# Example usage
population_size = 100
n = 8  # Number of queens
crossover_rate = 0.8
mutation_rate = 0.1
generations = 1000

# Create NQueensGeneticAlgorithm object
ga = NQueensGeneticAlgorithm(population_size, n, crossover_rate, mutation_rate)
# Solve N-Queens problem
solution = ga.evolve(generations)
print("Best solution:", solution)


# # 2 

import random

def initialize_population(pop_size, n):
    return [[random.randint(0, n-1) for _ in range(n)] for _ in range(pop_size)]

def fitness(chromosome):
    n = len(chromosome)
    attacks = 0
    for i in range(n):
        for j in range(i+1, n):
            if chromosome[i] == chromosome[j] or abs(chromosome[i] - chromosome[j]) == abs(i - j):
                attacks += 1
    return n*(n-1)//2 - attacks

def crossover_uniform(parent1, parent2):
    n = len(parent1)
    child = [-1] * n
    for i in range(n):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    return child

def crossover_shuffle(parent1, parent2):
    n = len(parent1)
    crossover_point = random.randint(1, n-1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def crossover_multiparent(parents):
    n = len(parents[0])
    child = [-1] * n
    for i in range(n):
        child[i] = random.choice(parents)[i]
    return child

def crossover_three_parent(parent1, parent2, parent3):
    n = len(parent1)
    child = [-1] * n
    for i in range(n):
        choices = [parent1[i], parent2[i], parent3[i]]
        child[i] = random.choice(choices)
    return child

def genetic_algorithm(pop_size, n, max_generations):
    population = initialize_population(pop_size, n)
    for generation in range(max_generations):
        # Evaluate fitness
        fitness_scores = [fitness(chromosome) for chromosome in population]
        # Selection
        selected_parents = random.choices(population, weights=fitness_scores, k=pop_size)
        # Crossover
        new_population = []
        for i in range(0, pop_size, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i+1]
            # Choose crossover technique
            child = crossover_uniform(parent1, parent2)
            #child = crossover_shuffle(parent1, parent2)
            #child = crossover_multiparent(selected_parents[i:i+3])
            #child = crossover_three_parent(parent1, parent2, selected_parents[i+2])
            new_population.append(child)
        population = new_population
    # Return the best solution found
    best_solution = max(population, key=fitness)
    return best_solution

# Example usage
n = 8  # Size of the board
pop_size = 100  # Population size
max_generations = 1000  # Maximum number of generations
solution = genetic_algorithm(pop_size, n, max_generations)
print("Solution:", solution)

# 3a TSP using GA and cyclic crossover inversion

import random
import numpy as np

class TSPGeneticAlgorithm:
    def __init__(self, num_cities, population_size, max_generations, cities_distances):
        self.num_cities = num_cities
        self.population_size = population_size
        self.max_generations = max_generations
        self.cities_distances = cities_distances

    def initialize_population(self):
        return [random.sample(range(self.num_cities), self.num_cities) for _ in range(self.population_size)]

    def rank_selection(self, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)
        ranks = np.arange(1, len(fitness_scores) + 1)
        probabilities = ranks / sum(ranks)
        selected_indices = np.random.choice(sorted_indices, size=self.population_size, replace=True, p=probabilities)
        return [self.population[i] for i in selected_indices]

    def calculate_fitness(self, chromosome):
        total_distance = sum(self.cities_distances[chromosome[i]][chromosome[(i + 1) % self.num_cities]] for i in range(self.num_cities))
        return 1 / total_distance  # Inverse of the total distance is used as fitness

    def crossover_cyclic_inversion(self, parent1, parent2):
        child = [-1] * self.num_cities
        start_index = random.randint(0, self.num_cities - 1)
        end_index = random.randint(0, self.num_cities - 1)
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        child[start_index:end_index + 1] = parent1[start_index:end_index + 1]
        for i in range(start_index, end_index + 1):
            if parent2[i] not in child:
                idx = i
                while True:
                    if parent2[idx] in child:
                        idx = parent2.index(parent2[idx])
                    else:
                        break
                child[idx] = parent2[i]
        for i in range(self.num_cities):
            if child[i] == -1:
                child[i] = parent2[i]
        return child

    def mutation_inversion_reversing(self, chromosome):
        start_index = random.randint(0, self.num_cities - 1)
        end_index = random.randint(0, self.num_cities - 1)
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        chromosome[start_index:end_index + 1] = chromosome[start_index:end_index + 1][::-1]
        return chromosome

    def evolve(self):
        self.population = self.initialize_population()
        for generation in range(self.max_generations):
            fitness_scores = [self.calculate_fitness(chromosome) for chromosome in self.population]
            # Selection
            selected_parents = self.rank_selection(fitness_scores)
            # Crossover
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1]
                child = self.crossover_cyclic_inversion(parent1, parent2)
                new_population.append(child)
            # Mutation
            for i in range(len(new_population)):
                if random.random() < mutation_rate:
                    new_population[i] = self.mutation_inversion_reversing(new_population[i])
            self.population = new_population
        best_solution = max(self.population, key=self.calculate_fitness)
        return best_solution

# Example usage
num_cities = 10  # Number of cities
population_size = 100  # Population size
max_generations = 1000  # Maximum number of generations
mutation_rate = 0.1  # Mutation rate
# Generate random distances between cities (example)
cities_distances = np.random.rand(num_cities, num_cities)
# Create TSP genetic algorithm object
tsp_genetic_algorithm = TSPGeneticAlgorithm(num_cities, population_size, max_generations, cities_distances)
# Solve TSP
best_route = tsp_genetic_algorithm.evolve()
print("Best Route:", best_route)

# 3b  TSP using partially mapped and scrambled mutn
import random
import numpy as np

class TSPGeneticAlgorithm:
    def __init__(self, num_cities, population_size, max_generations, cities_distances):
        self.num_cities = num_cities
        self.population_size = population_size
        self.max_generations = max_generations
        self.cities_distances = cities_distances

    def initialize_population(self):
        return [random.sample(range(self.num_cities), self.num_cities) for _ in range(self.population_size)]

    def calculate_fitness(self, chromosome):
        total_distance = sum(self.cities_distances[chromosome[i]][chromosome[(i + 1) % self.num_cities]] for i in range(self.num_cities))
        return 1 / total_distance  # Inverse of the total distance is used as fitness

    def tournament_selection(self, population, k=3):
        participants = random.sample(population, k)
        return max(participants, key=self.calculate_fitness)

    def partially_mapped_crossover(self, parent1, parent2):
        # Randomly select two crossover points
        start_idx = random.randint(0, self.num_cities - 1)
        end_idx = random.randint(start_idx, self.num_cities - 1)
        # Create child
        child = [None] * self.num_cities
        for i in range(start_idx, end_idx + 1):
            child[i] = parent1[i]
        # Map elements from parent2 to child
        for i in range(start_idx, end_idx + 1):
            if parent2[i] not in child:
                idx = parent2.index(parent1[i])
                while child[idx] is not None:
                    idx = parent2.index(parent1[idx])
                child[idx] = parent2[i]
        # Copy the remaining elements from parent2
        for i in range(self.num_cities):
            if child[i] is None:
                child[i] = parent2[i]
        return child

    def scrambled_mutation(self, chromosome):
        # Randomly select two positions and scramble the values in between
        start_idx = random.randint(0, self.num_cities - 1)
        end_idx = random.randint(start_idx, self.num_cities - 1)
        scrambled_section = chromosome[start_idx:end_idx + 1]
        random.shuffle(scrambled_section)
        chromosome[start_idx:end_idx + 1] = scrambled_section
        return chromosome

    def evolve(self):
        self.population = self.initialize_population()
        for generation in range(self.max_generations):
            fitness_scores = [self.calculate_fitness(chromosome) for chromosome in self.population]
            # Selection
            selected_parents = [self.tournament_selection(self.population) for _ in range(self.population_size)]
            # Crossover
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1]
                child = self.partially_mapped_crossover(parent1, parent2)
                new_population.append(child)
            # Mutation
            for i in range(len(new_population)):
                if random.random() < mutation_rate:
                    new_population[i] = self.scrambled_mutation(new_population[i])
            self.population = new_population
        best_solution = max(self.population, key=self.calculate_fitness)
        return best_solution

# Example usage
num_cities = 10  # Number of cities
population_size = 100  # Population size
max_generations = 10  # Maximum number of generations
mutation_rate = 0.1  # Mutation rate
# Generate random distances between cities (example)
cities_distances = np.random.rand(num_cities, num_cities)
# Create TSP genetic algorithm object
tsp_genetic_algorithm = TSPGeneticAlgorithm(num_cities, population_size, max_generations, cities_distances)
# Solve TSP
best_route = tsp_genetic_algorithm.evolve()
print("Best Route:", best_route)

# 5  ACO job shop
import numpy as np

class AntColony:
    def __init__(self, num_ants, num_iterations, num_jobs, num_machines, pheromone_init=0.1,
                 alpha=1, beta=2, evaporation_rate=0.5):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.pheromone_init = pheromone_init
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        # Initialize pheromone matrix
        self.pheromones = np.full((num_jobs, num_jobs), pheromone_init)

    def run(self):
        for iteration in range(self.num_iterations):
            solutions = []
            for ant in range(self.num_ants):
                solution = self.construct_solution()
                solutions.append((solution, self.calculate_cost(solution)))
            # Update pheromones
            self.update_pheromones(solutions)
            # Evaporate pheromones
            self.pheromones *= self.evaporation_rate
            # Choose best solution
            best_solution = min(solutions, key=lambda x: x[1])[0]
            print(f"Iteration {iteration}: Best Solution: {best_solution}")

    def construct_solution(self):
        solution = []
        for job in range(self.num_jobs):
            machine = np.random.randint(0, self.num_machines)
            solution.append((job, machine))
        return solution

    def calculate_cost(self, solution):
        # Placeholder cost function, to be replaced with actual cost calculation
        return np.random.rand()

    def update_pheromones(self, solutions):
        for solution, cost in solutions:
            for job, machine in solution:
                self.pheromones[job, machine] += 1 / cost

if __name__ == "__main__":
    num_ants = 10
    num_iterations = 20
    num_jobs = 5
    num_machines = 3
    ant_colony = AntColony(num_ants, num_iterations, num_jobs, num_machines)
    ant_colony.run()

# 6 PSO 
import numpy as np

def objective_function(a):
    return 1 + 2*a[0] + (3*a[1] - 1) + 3*a[2] + 2*a[3]**2 + (5*a[4] + 2)

n_particles = 20
n_dimensions = 5
max_iter = 50
w_max = 0.9
w_min = 0.3
c1 = 1
c2 = 1
bounds = [(10, 60), (15, 30), (27, 75), (10, 30), (10, 50)]

positions = np.random.uniform(low=np.array([bound[0] for bound in bounds]),
                              high=np.array([bound[1] for bound in bounds]),
                              size=(n_particles, n_dimensions))
velocities = np.random.uniform(low=-1, high=1, size=(n_particles, n_dimensions))

# Initialize best positions and global best
best_positions = positions.copy()
best_values = np.array([objective_function(pos) for pos in positions])
global_best_index = np.argmax(best_values)
global_position = positions[global_best_index].copy()
global_best_value = best_values[global_best_index]

# Perform PSO iterations
for _ in range(max_iter):
    w = np.random.uniform(low=w_min, high=w_max)
    # w = w_max - (_ / max_iter) * (w_max - w_min)
    r1 = np.random.uniform(size=(n_particles, n_dimensions))
    r2 = np.random.uniform(size=(n_particles, n_dimensions))
    
    velocities = (w * velocities + 
                  c1 * r1 * (best_positions - positions) + 
                  c2 * r2 * (global_position - positions))
    positions += velocities
    
    # Apply bounds
    positions = np.clip(positions, 
                        np.array([bound[0] for bound in bounds]), 
                        np.array([bound[1] for bound in bounds]))
    
    # Update best positions
    current_values = np.array([objective_function(pos) for pos in positions])
    for i in range(n_particles):
        if current_values[i] > best_values[i]:
            best_positions[i] = positions[i].copy()
            best_values[i] = current_values[i]
    
    new_global_best_index = np.argmax(best_values)
    if best_values[new_global_best_index] > global_best_value:
        global_best_index = new_global_best_index
        global_position = best_positions[global_best_index].copy()
        global_best_value = best_values[global_best_index]

print("Optimal solution:")
print("a1 =", global_position[0])
print("a2 =", global_position[1])
print("a3 =", global_position[2])
print("a4 =", global_position[3])
print("a5 =", global_position[4])
print("Maximum value of f(a1,a2,a3,a4,a5) =", global_best_value)
