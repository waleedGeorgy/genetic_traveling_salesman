import random
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import pygad
from geopy.distance import geodesic

# read in data and check column names 
data = pd.read_csv('starbucks.csv')
print(data.columns)

df = data[data['countryCode'] == 'GB']
df.reset_index(inplace=True)

# check for invalid lon/lat pairs
len(df.dropna(subset=['latitude', 'longitude'])) - len(df)

# ## 2. Exploratory analysis

# ### 2.1 Distribution of cafés by town

vis = df.groupby('city').storeNumber.count().reset_index()
px.bar(vis, x='city', y='storeNumber', template='seaborn')

# ### 2.2 Map of cafés in the UK

full_map = folium.Map(location=[51.509685, -0.118092], zoom_start=6, tiles="stamentoner")

for _, r in df.iterrows():
    folium.Marker([r['latitude'], r['longitude']], popup=f'<i>{r["storeNumber"]}</i>').add_to(full_map)

full_map.save('main_map.html')

# ## 3. Testing the distance methodology

# To assess how good each solution is there needs to be a measure of fitness.
# For the purpose of this example the distance 'as the crow flies' is used
# without taking into account actual road distances however this could be explored in the future.

origin = (df['latitude'][0], df['longitude'][0])
dest = (df['latitude'][100], df['longitude'][100])

print(geodesic(origin, dest).kilometers)

# ## 4. Preparing data structures

# The data structures needed for testing solutions are the "genes" or store options to select from named *genes*

# A lookup to access these genes known as *stores* 

# A *check_range* which is used to check that every option is given in a solution (a key criteria in the TSP).

test = df.head(10)
genes = {store_num: [lat, lon] for store_num, lat, lon in zip(test['storeNumber'], test['latitude'], test['longitude'])}
stores = list(genes.keys())
check_range = [i for i in range(0, 10)]


# ## 5. Defining functions 
# 
# The algorithm requires a set of functions to be pre-defined as the out of the box genetic algorithm does not support a TSP.
# 
#  1. build_population: builds a population of chromosomes to test with proper restrictions applied
#  2. fitness_func: Used to test a solution to see how well it performs, in this case the fitness_func will be assessed based on the distance as the crow flies between each successive point
#  3. pmx_crossover: performs the crossover of a parent and child with proper Partially Matched Crossover (PMX) logic
#  4. crossover_func: applies the crossover
#  5. on_crossover: applies the mutation after crossover
#  6. on_generation: used to print the progress and results at each generation

# Assess the quality or fitness of a solution so that only the fittest are selected for the next generation and to breed.

def build_population(size, chromosome_size):
    population = []
    for i in range(size):
        home_city = 0
        added = {home_city: 'Added'}
        chromosome = [home_city]

        while len(chromosome) < chromosome_size:
            proposed_gene = random.randint(0, chromosome_size - 1)
            if added.get(proposed_gene) is None:
                chromosome.append(proposed_gene)
                added.update({proposed_gene: 'Added'})
            else:
                pass

        chromosome.append(home_city)

        population.append(chromosome)

    return np.array(population)


population = build_population(100, 10)
print(population.shape)


def fitness_func(solution, solution_idx):
    # loop through the length of the chromosome finding the distance between each
    # gene added

    #  to increment
    total_dist = 0

    for gene in range(0, len(solution)):

        # get the lon lat of the two points
        a = genes.get(stores[solution[gene]])

        try:
            b = genes.get(stores[solution[gene + 1]])

            # find the distance (crow flies)
            dist = geodesic(a, b).kilometers

        except IndexError:
            dist = 0

        total_dist += dist

    # to optimise this value in the positive direction the inverse of dist is used
    fitness = 1 / total_dist

    return fitness


def pmx_crossover(parent1, parent2, sequence_start, sequence_end):
    # initialise a child
    child = np.zeros(parent1.shape[0])

    # get the genes for parent one that are passed on to child one
    parent1_to_child1_genes = parent1[sequence_start:sequence_end]

    # get the position of genes for each respective combination
    parent1_to_child1 = np.isin(parent1, parent1_to_child1_genes).nonzero()[0]

    for gene in parent1_to_child1:
        child[gene] = parent1[gene]

    # gene of parent 2 not in the child
    genes_not_in_child = parent2[np.isin(parent2, parent1_to_child1_genes, invert=True).nonzero()[0]]

    # if the gene is not already
    if genes_not_in_child.shape[0] >= 1:
        for gene in genes_not_in_child:
            if gene >= 1:
                lookup = gene
                not_in_sequence = True

                while not_in_sequence:
                    position_in_parent2 = np.where(parent2 == lookup)[0][0]

                    if position_in_parent2 in range(sequence_start, sequence_end):
                        lookup = parent1[position_in_parent2]

                    else:
                        child[position_in_parent2] = gene
                        not_in_sequence = False

    return child


def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        # locate the parents
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        # find gene sequence in parent 1
        sequence_start = random.randint(1, parent1.shape[0] - 4)
        sequence_end = random.randint(sequence_start, parent1.shape[0] - 1)

        # perform crossover
        child1 = pmx_crossover(parent1, parent2, sequence_start, sequence_end)
        child2 = pmx_crossover(parent2, parent1, sequence_start, sequence_end)

        offspring.append(child1)
        offspring.append(child2)

        idx += 1

    return np.array(offspring)


# The mutation function chosen is inversion as it does not invalidate the solution.

def mutation_func(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        # define a sequence of genes to reverse
        sequence_start = random.randint(1, offspring[chromosome_idx].shape[0] - 2)
        sequence_end = random.randint(sequence_start, offspring[chromosome_idx].shape[0] - 1)

        genes = offspring[chromosome_idx, sequence_start:sequence_end]

        # start at the start of the sequence assigning the reverse sequence back to the chromosome
        index = 0
        if len(genes) > 0:
            for gene in range(sequence_start, sequence_end):
                offspring[chromosome_idx, gene] = genes[index]

                index += 1

        return offspring


# Used in the genetic algorithm flow to apply the custom mutation after crossover

def on_crossover(ga_instance, offspring_crossover):
    # apply mutation to ensure uniqueness 
    offspring_mutation = mutation_func(offspring_crossover, ga_instance)

    # save the new offspring set as the parents of the next generation
    ga_instance.last_generation_offspring_mutation = offspring_mutation


# Added for debugging and assessing progress by generation at runtime

def on_generation(ga):
    print("Generation", ga.generations_completed)
    print(ga.population)


# ## 6. Executing the algorithm
# The genetic algorithm is set up as instance and at initialisation several parameters are given.
# The algorithm then runs to find the best solution for a set number of generations.

# ### 6.1 Example Initialising the algorithm
# 
# Notable parameters include:
#   - The use of gene space to limit the possible genes chosen to just be those in the TSP range
#   - Mutations being turned off temporarily
#   - Implementation of custom on_ functions 
#   - Allow duplication of genes parameter set to false to ensure any newly
#     introduced chromosomes/chromosomes created as population is initialised have no duplicate genes

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=40,
                       fitness_func=fitness_func,
                       sol_per_pop=200,
                       initial_population=population,
                       gene_space=range(0, 10),
                       gene_type=int,
                       mutation_type=mutation_func,
                       on_generation=on_generation,
                       crossover_type=crossover_func,
                       keep_parents=6,
                       mutation_probability=0.4)

# 6.2 Running the algorithm
# The genetic algorithm is run with a simple function call

ga_instance.run()

# ## 7. Assessing results
# The result solution can be checked and analysed using the ga_instance itself

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f'Generation of best solution: {ga_instance.best_solution_generation}')
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format
          (best_solution_generation=ga_instance.best_solution_generation))


# 7.1 Verifying a solution

# For a solution to be valid it needs to have:
#  - A maximum gene value that matches the total number of stores 
#  - A minimum gene value of 0 
#  - Each gene must be unique

def verify_solution(solution, max_gene):
    if min(solution) != 0:
        print('Failed values below 0')

    if max(solution) != max_gene:
        print('Failed values less than or above max possible value')

    if len(set(solution)) - len(solution) != -1:
        print(len(set(solution)) - len(solution))
        print('Failed solution does not contain unique values')


verify_solution(solution, 9)

print(solution)

# ### 7.2 Interpreting the result sequence can be used to access latitude and longitude for each store in the solution.

points = [genes.get(stores[id]) + [stores[id]] for id in solution]
print(points[:5])

shortest_map = folium.Map(location=[51.509685, -0.118092], zoom_start=6, tiles="stamentoner")

for point in range(0, len(points)):
    folium.Marker([points[point][0], points[point][1]], popup=f'<i>{points[point][2]}</i>').add_to(shortest_map)

    try:
        folium.PolyLine([(points[point][0], points[point][1]), (points[point + 1][0], points[point + 1][1])],
                        color='red', weight=5, opacity=0.8).add_to(shortest_map)

    except IndexError:
        pass

shortest_map.save('shortest_map.html')

# The map shows the shortest path that has been found. So that the travelling coffee drinker can maximise the time on coffee and minimise the time on travelling.
# 
# Now the algorithm can be scaled up for the whole of the UK, or tailored to just one town. An example of the solution scaled to the UK is given below.

# ## 8. Scaling up the solution

df = df[df['city'] == 'London']
genes = {store_num: [lat, lon] for store_num, lat, lon in zip(df['storeNumber'], df['latitude'], df['longitude'])}
stores = list(genes.keys())
len(stores)

population = build_population(200, 165)
len(population[0])


# ### 8.1 Building the final algorithm
# The code to build the algorithm has to be re-run with the above data structures altered.

def fitness_func(solution, solution_idx):
    # loop through the length of the chromosome finding the distance between each
    total_dist = 0

    for gene in range(0, len(solution)):

        # get the lon lat of the two points
        a = genes.get(stores[solution[gene]])

        try:
            b = genes.get(stores[solution[gene + 1]])

            # find the distance (crow flies)
            dist = geodesic(a, b).kilometers

        except IndexError:
            dist = 0

        total_dist += dist

    # to optimise this value in the positive direction the inverse of dist is used
    fitness = 1 / total_dist

    return fitness


def pmx_crossover(parent1, parent2, sequence_start, sequence_end):
    # initialise a child
    child = np.zeros(parent1.shape[0])

    # get the genes for parent one that are passed on to child one
    parent1_to_child1_genes = parent1[sequence_start:sequence_end]

    # get the position of genes for each respective combination
    parent1_to_child1 = np.isin(parent1, parent1_to_child1_genes).nonzero()[0]

    for gene in parent1_to_child1:
        child[gene] = parent1[gene]

    # gene of parent 2 not in the child
    genes_not_in_child = parent2[np.isin(parent2, parent1_to_child1_genes, invert=True).nonzero()[0]]

    if genes_not_in_child.shape[0] >= 1:
        for gene in genes_not_in_child:
            if gene >= 1:
                lookup = gene
                not_in_sequence = True

                while not_in_sequence:
                    position_in_parent2 = np.where(parent2 == lookup)[0][0]

                    if position_in_parent2 in range(sequence_start, sequence_end):
                        lookup = parent1[position_in_parent2]

                    else:
                        child[position_in_parent2] = gene
                        not_in_sequence = False

    return child


def crossover_func(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        # locate the parents
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        # find gene sequence in parent 1
        sequence_start = random.randint(1, parent1.shape[0] - 4)
        sequence_end = random.randint(sequence_start, parent1.shape[0] - 1)

        # perform crossover
        child1 = pmx_crossover(parent1, parent2, sequence_start, sequence_end)
        child2 = pmx_crossover(parent2, parent1, sequence_start, sequence_end)

        offspring.append(child1)
        offspring.append(child2)

        idx += 1

    return np.array(offspring)


def mutation_func(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):
        # define a sequence of genes to reverse
        sequence_start = random.randint(1, offspring[chromosome_idx].shape[0] - 2)
        sequence_end = random.randint(sequence_start, offspring[chromosome_idx].shape[0] - 1)

        genes = offspring[chromosome_idx, sequence_start:sequence_end]

        # start at the start of the sequence assigning the reverse sequence back to the chromosome
        index = 0
        if len(genes) > 0:
            for gene in range(sequence_start, sequence_end):
                offspring[chromosome_idx, gene] = genes[index]

                index += 1

        return offspring


def on_crossover(ga_instance, offspring_crossover):
    # apply mutation to ensure uniqueness 
    offspring_mutation = mutation_func(offspring_crossover, ga_instance)

    # save the new offspring set as the parents of the next generation
    ga_instance.last_generation_offspring_mutation = offspring_mutation


def on_generation(ga):
    print("Generation", ga.generations_completed)
    print(ga.population)


ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=40,
                       fitness_func=fitness_func,
                       sol_per_pop=200,
                       initial_population=population,
                       gene_space=range(0, 165),
                       gene_type=int,
                       mutation_type=mutation_func,
                       on_generation=on_generation,
                       crossover_type=crossover_func,
                       keep_parents=6,
                       mutation_probability=0.4)

ga_instance.run()

# ## 8.2 Evaluating the final algorithm

# The overall solution can now be assessed.

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f'Generation of best solution: {ga_instance.best_solution_generation}')
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

verify_solution(solution, len(stores))
print(solution)

points = [genes.get(stores[id]) + [stores[id]] for id in solution]
print(points[:5])

final_map = folium.Map(location=[51.509685, -0.118092], zoom_start=6, tiles="stamentoner")

for point in range(0, len(points)):
    folium.Marker([points[point][0], points[point][1]], popup=f'<i>{points[point][2]}</i>').add_to(final_map)

    try:
        folium.PolyLine([(points[point][0], points[point][1]),
                         (points[point + 1][0], points[point + 1][1])],
                        color='red',
                        weight=5,
                        opacity=0.8).add_to(final_map)

    except IndexError:
        pass

final_map.save('final_map.html')


# ## 10. Total result

# The total resulting distance around London after optimising the solution is:

def distance(solution):
    # loop through the length of the chromosome finding the distance between each
    total_dist = 0

    for gene in range(0, len(solution)):

        # get the lon lat of the two points
        a = genes.get(stores[solution[gene]])

        try:
            b = genes.get(stores[solution[gene + 1]])

            # find the distance (crow flies)
            dist = geodesic(a, b).kilometers

        except IndexError:
            dist = 0

        total_dist += dist

    # to optimise this value in the positive direction the inverse of dist is used

    return total_dist


print(f'Total distance traveled = {distance(solution)}')
