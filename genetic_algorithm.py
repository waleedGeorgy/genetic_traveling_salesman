import random
import folium
import numpy as np
import pandas as pd
import plotly.express
import pygad
from geopy.distance import geodesic

# База данных сети кафе Старбакс по ссылке https://www.kaggle.com/datasets/starbucks/store-locations
data = pd.read_csv('sb_moscow.csv')
print(data.columns)

# Выдиляем только кафе в России
df = data[data['countryCode'] == 'RU']
df.reset_index(inplace=True)

len(df.dropna(subset=['latitude', 'longitude'])) - len(df)

city_visited = df.groupby('city').storeNumber.count().reset_index()
plotly.express.bar(city_visited, x='city', y='storeNumber', template='seaborn')

full_map = folium.Map(location=[55.755825, 37.617298], zoom_start=10)

for _, r in df.iterrows():
    folium.Marker([r['latitude'], r['longitude']], popup=f'<i>{r["storeNumber"]}</i>').add_to(full_map)

# Выводится карта с местами положения всех найденных кафе
full_map.save('main_map.html')


# Функция построения начальной популяции
def build_the_population(size, chromosome_size):
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


# Проверка полученного решения, Для того чтобы решение было действительным, оно должно иметь:
# Минимальное значение гена, которое должно быть равно нулю.
# Максимальное значение гена, которое соответствует общему количеству кафе.
# Каждый ген в решении должен быть уникальным.

def solution_verification(solution, max_gene):
    if min(solution) != 0:
        print('Minimum gene value not equal to 0')

    if max(solution) != max_gene:
        print('Maximum gene value is not equal to desired solution')

    if len(set(solution)) - len(solution) != -1:
        print(len(set(solution)) - len(solution))
        print('Not every gene in the solution is unique')


# Выделяем только кафе в москве и московской области
df = df[df['city'] == 'Moscow']
genes = {store_num: [lat, lon] for store_num, lat, lon in zip(df['storeNumber'], df['latitude'], df['longitude'])}
stores = list(genes.keys())
len(stores)

population = build_the_population(200, len(stores))
len(population[0])


# Функция пригодности, для определения самых пригодных хромосом

def fitness_function(solution, solution_idx):
    total_dist = 0

    for gene in range(0, len(solution)):

        a = genes.get(stores[solution[gene]])

        try:
            b = genes.get(stores[solution[gene + 1]])
            dist = geodesic(a, b).kilometers

        except IndexError:
            dist = 0

        total_dist += dist

    fitness = 1 / total_dist

    return fitness


# Функция определения метода частичного согласованного скрещивания (PMX).
def pmx_crossover(parent1, parent2, start_of_sequence, end_of_sequence):
    child = np.zeros(parent1.shape[0])

    transfer_genes = parent1[start_of_sequence:end_of_sequence]

    parent1_child1 = np.isin(parent1, transfer_genes).nonzero()[0]

    for gene in parent1_child1:
        child[gene] = parent1[gene]

    genes_not_in_child = parent2[np.isin(parent2, transfer_genes, invert=True).nonzero()[0]]

    if genes_not_in_child.shape[0] >= 1:
        for gene in genes_not_in_child:
            if gene >= 1:
                lookup = gene
                not_in_sequence = True

                while not_in_sequence:
                    parent2_position = np.where(parent2 == lookup)[0][0]

                    if parent2_position in range(start_of_sequence, end_of_sequence):
                        lookup = parent1[parent2_position]

                    else:
                        child[parent2_position] = gene
                        not_in_sequence = False

    return child


# Функция процесса скрещивания
def crossover_function(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        start_of_sequence = random.randint(1, parent1.shape[0] - 4)
        end_of_sequence = random.randint(start_of_sequence, parent1.shape[0] - 1)

        child1 = pmx_crossover(parent1, parent2, start_of_sequence, end_of_sequence)
        child2 = pmx_crossover(parent2, parent1, start_of_sequence, end_of_sequence)

        offspring.append(child1)
        offspring.append(child2)

        idx += 1

    return np.array(offspring)


# Функция мутации
def mutation_function(offspring, ga_instance):
    for chromosome_idx in range(offspring.shape[0]):

        start_of_sequence = random.randint(1, offspring[chromosome_idx].shape[0] - 2)
        end_of_sequence = random.randint(start_of_sequence, offspring[chromosome_idx].shape[0] - 1)

        genes = offspring[chromosome_idx, start_of_sequence:end_of_sequence]

        index = 0
        if len(genes) > 0:
            for gene in range(start_of_sequence, end_of_sequence):
                offspring[chromosome_idx, gene] = genes[index]

                index += 1

        return offspring


def on_crossover(ga_instance, offspring_crossover):
    offspring_mutation = mutation_function(offspring_crossover, ga_instance)

    ga_instance.last_generation_offspring_mutation = offspring_mutation


def on_generation(ga):
    print("Generation", ga.generations_completed)
    print(ga.population)


# Переменная генетического алгоритма, которая имеет значения всех определенных выше функций
ga_instance = pygad.GA(num_generations=100, num_parents_mating=40,
                       fitness_func=fitness_function, sol_per_pop=200, initial_population=population,
                       gene_space=range(0, 100), gene_type=int, mutation_type=mutation_function,
                       on_generation=on_generation, crossover_type=crossover_function,
                       keep_parents=6, mutation_probability=0.4)

ga_instance.run()

# Вывод поколения, значения пригодности, и индекс лучшего решения
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f'Generation of best solution: {ga_instance.best_solution_generation}')
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

solution_verification(solution, len(stores)-1)
# Хромосома лучшего решения
print(f'Solution chromosome: {solution}')

points = [genes.get(stores[id]) + [stores[id]] for id in solution]

final_map = folium.Map(location=[55.755825, 37.617298], zoom_start=10)

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
# Карта на которой выводится маршрут коммивояжера для хромосомы решения
final_map.save('final_map.html')


# Функция для вывода общего пройденного расстояния
def distance_traveled(solution):
    total_dist = 0

    for gene in range(0, len(solution)):
        a = genes.get(stores[solution[gene]])

        try:
            b = genes.get(stores[solution[gene + 1]])
            dist = geodesic(a, b).kilometers

        except IndexError:
            dist = 0

        total_dist += dist

    return total_dist


print(f'Overall distance traveled = {distance_traveled(solution)}')
