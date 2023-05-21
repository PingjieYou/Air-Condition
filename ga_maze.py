from random import randint, random, choice
from pyamaze import maze, agent
from copy import deepcopy, copy
import csv

# initialize constants
POPULATIONSIZE = 100
ROWS, COLS = 10, 10

# Set the mutation rate
MUTATION_RATE = 100

# start and end points of the maze
START, END = 1, COLS

# weights of path length, infeasible steps and number of turns
wl, ws, wt = 2, 3, 2

# file name for storing fitness parameters
file_name = 'data.csv'

# initilize a maze object and create a maze
m = maze(ROWS, COLS)
m.CreateMaze(loopPercent=100)
maps = m.maze_map


def popFiller(pop, direction):
    """
    Takes in population and direction lists and fills them with random values within the range.
    """
    for i in range(POPULATIONSIZE):
        # Generate a random path and add it to the population list
        pop.append([(randint(1, ROWS), j) for j in range(1, COLS + 1)])
        # Set the start and end points of the path
        pop[i][0] = (START, START)
        pop[i][-1] = (ROWS, END)
        # Generate a random direction and add it to the direction list
        direction.append(choice(["r", "c"]))
    return pop, direction


def inter_steps(point1, point2, direction):
    """
    Takes in two points and the direction of the path between them and returns the intermediate steps between them.
    """
    steps = []
    if direction == "c":  # column first
        if point1[0] < point2[0]:
            steps.extend([(i, point1[1]) for i in range(point1[0] + 1, point2[0] + 1)])
        elif point1[0] > point2[0]:
            steps.extend([(i, point1[1]) for i in range(point1[0] - 1, point2[0] - 1, -1)])
        steps.append(point2)
    elif direction == "r":  # row first
        if point1[0] < point2[0]:
            steps.extend([(i, point1[1] + 1) for i in range(point1[0], point2[0] + 1)])
        elif point1[0] > point2[0]:
            steps.extend([(i, point1[1] + 1) for i in range(point1[0], point2[0] - 1, -1)])
        else:
            steps.append(point2)
    return steps


def path(individual, direction):
    """
    Takes in the population list and the direction list and returns the complete path using the inter_steps function.
    """
    complete_path = [individual[0]]
    for i in range(len(individual) - 1):
        # Get the intermediate steps between the current and the next point in the population
        complete_path.extend(inter_steps(individual[i], individual[i + 1], direction))
    return complete_path


def pathParameters(individual, complete_path, map_data):
    """
    Takes in an individual, it's complete path and the map data
    and returns the number of turns and the number of infeasible steps.
    """
    # Count the number of turns in the individual's path
    turns = sum(
        1
        for i in range(len(individual) - 1)
        if individual[i][0] != individual[i + 1][0]
    )
    # Count the number of Infeasible steps in the individual's path
    infeas = sum(
        any(
            (
                map_data[complete_path[i]]["E"] == 0 and complete_path[i][1] == complete_path[i + 1][1] - 1,
                map_data[complete_path[i]]["W"] == 0 and complete_path[i][1] == complete_path[i + 1][1] + 1,
                map_data[complete_path[i]]["S"] == 0 and complete_path[i][0] == complete_path[i + 1][0] - 1,
                map_data[complete_path[i]]["N"] == 0 and complete_path[i][0] == complete_path[i + 1][0] + 1,
            )
        )
        for i in range(len(complete_path) - 1)
    )
    return turns, infeas


def fitCal(population, direction, solutions, createCSV=True):
    """
    Takes in the population list and the direction list and
    returns the fitness list of the population and the solution found(infeasible steps equal to zero).

    Args:
    - population (list): a list of individuals in the population
    - direction (list): a list of the direction for each individual
    - solutions (list): a list of the solutions found so far
    - createCSV (bool): a flag indicating whether to create a CSV file or not (default: True)
    """
    # Initialize empty lists for turns, infeasible steps, and path length
    turns = []
    infeas_steps = []
    length = []

    # A function for calculating the normalized fitness value
    def calc(curr, maxi, mini):
        if maxi == mini:
            return 0
        else:
            return 1 - ((curr - mini) / (maxi - mini))

    # Iterate over each individual in the population
    for i, individual in enumerate(population):
        # Generate the complete path of individual
        p = path(individual, direction[i])
        # Calculate the number of turns and infeasible steps in the individual's path
        t, s = pathParameters(individual, p, maps)
        # If the individual has zero infeasible steps, add it to the solutions list
        if s == 0:
            solutions.append([deepcopy(individual), copy(direction[i])])
        # Add the individual's number of turns, infeasible steps, and path length to their respective lists
        turns.append(t)
        infeas_steps.append(s)
        length.append(len(p))

    # Calculate the maximum and minimum values for turns, infeasible steps, and path length
    max_t, min_t = max(turns), min(turns)
    max_s, min_s = max(infeas_steps), min(infeas_steps)
    max_l, min_l = max(length), min(length)

    # Calculate the normalized fitness values for turns, infeasible steps, and path length
    fs = [calc(infeas_steps[i], max_s, min_s) for i in range(len(population))]
    fl = [calc(length[i], max_l, min_l) for i in range(len(population))]
    ft = [calc(turns[i], max_t, min_t) for i in range(len(population))]

    # Calculate the fitness scores for each individual in the population
    fitness = [
        (100 * ws * fs[i]) * ((wl * fl[i] + wt * ft[i]) / (wl + wt))
        for i in range(POPULATIONSIZE)
    ]

    # If createCSV flag is True, write the parameters of fitness to a CSV file
    if createCSV:
        with open(file_name, 'a+', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Path Length', 'Turns', 'Infeasible Steps', 'Fitness'])
            writer.writerow({'Path Length': min_l, 'Turns': min_t, 'Infeasible Steps': min_s, 'Fitness': max(fitness)})

    return fitness, solutions


def rankSel(population, fitness, direction):
    """
    Takes in the population, fitness and direction lists and returns the population and direction lists after rank selection.
    """
    # Pair each fitness value with its corresponding individual and direction
    pairs = zip(fitness, population, direction)
    # Sort pairs in descending order based on fitness value
    sorted_pair = sorted(pairs, key=lambda x: x[0], reverse=True)
    # Unzip the sorted pairs into separate lists
    _, population, direction = zip(*sorted_pair)

    return list(population), list(direction)


def crossover(population, direction):
    """
    Takes in the population and direction lists and returns the population and direction lists after single point crossover.
    """
    # Choose a random crossover point between the second and second-to-last gene
    crossover_point = randint(2, len(population[0]) - 2)
    # Calculate the number of parents to mate (half the population size)
    no_of_parents = POPULATIONSIZE // 2
    for i in range(no_of_parents - 1):
        # Create offspring by combining the genes of two parents up to the crossover point
        population[i + no_of_parents] = (
                population[i][:crossover_point] + population[i + 1][crossover_point:]
        )
        # Choose a random direction for the offspring
        direction[i + no_of_parents] = choice(["r", "c"])
    return population, direction


def mutation(population, mutation_rate, direction, no_of_genes_to_mutate=1):
    """
    Takes in the population, mutation rate, direction and number of genes to mutate
    and returns the population and direction lists after mutation.
    """
    # Validate the number of genes to mutate
    if no_of_genes_to_mutate <= 0:
        raise ValueError("Number of genes to mutate must be greater than 0")
    if no_of_genes_to_mutate > COLS:
        raise ValueError(
            "Number of genes to mutate must not be greater than number of columns"
        )

    for i in range(POPULATIONSIZE):
        # Check if the individual will be mutated based on the mutation rate
        if random() < mutation_rate:
            for _ in range(no_of_genes_to_mutate):
                # Choose a random gene to mutate and replace it with a new random gene
                index = randint(1, COLS - 2)
                population[i][index] = (randint(1, ROWS), population[i][index][1])
            # Choose a random direction for the mutated individual
            direction[i] = choice(["r", "c"])
    return population, direction


def best_solution(solutions):
    """Takes a list of solutions and returns the best individual(list) and direction"""
    # Initialize the best individual and direction as the first solution in the list
    best_individual, best_direction = solutions[0]
    # Calculate the length of the path for the best solution
    min_length = len(path(best_individual, best_direction))

    for individual, direction in solutions[1:]:
        # Calculate the length of the path for the best solution
        current_length = len(path(individual, direction))
        # If the current solution is better than the best solution, update the best solution
        if current_length < min_length:
            min_length = current_length
            best_individual = individual
            best_direction = direction
    return best_individual, best_direction


def main():
    # Initialize population, direction, and solution lists
    pop, direc, sol = [], [], []
    # Set the generation count and the maximum number of generations
    gen, maxGen = 0, 2000
    # Set the number of solutions to be found
    sol_count = 1
    # Set the number of solutions to be found
    pop, direc = popFiller(pop, direc)

    # Create a new CSV file and write header
    with open(file_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Path Length', 'Turns', 'Infeasible Steps', 'Fitness'])
        writer.writeheader()

    # Start the loop for the genetic algorithm
    print('Running...')
    while True:
        gen += 1
        # Calculate the fitness values for the population and identify any solutions
        fitness, sol = fitCal(pop, direc, sol, createCSV=True)
        # Select the parents for the next generation using rank selection
        pop, direc = rankSel(pop, fitness, direc)
        # Create the offspring for the next generation using crossover
        pop, direc = crossover(pop, direc)
        # mutate the offsprings using mutation function
        pop, direc = mutation(pop, MUTATION_RATE, direc, no_of_genes_to_mutate=1)
        # Check if the required number of solutions have been found
        if len(sol) == sol_count:
            print("Solution found!")
            break

        # Check if the maximum number of generations have been reached
        if gen >= maxGen:
            print("Solution not found!")
            flag = input("Do you  want to create another population(y/n): ")
            # if flag is 'y', clear the population and direction lists and refill them
            if flag == 'y':
                pop, direc = [], []
                pop, direc = popFiller(pop, direc)
                gen = 0
                continue
            # If flag is 'n' exit the program
            else:
                print("Good Bye")
                return None

                # Find the best solution and its direction
    solIndiv, solDir = best_solution(sol)
    # Generate the final solution path and reverse it
    solPath = path(solIndiv, solDir)
    solPath.reverse()

    # Create an agent, set its properties, and trace its path through the maze
    a = agent(m, shape="square", filled=False, footprints=True)
    m.tracePath({a: solPath}, delay=100)
    m.run()


if __name__ == "__main__":
    # Call the main function
    main()

    # if you want to time the main program (Note: You should comment out the line 289-291 for the proper timing of the code)
    # start_time = time.time()
    # main()
    # print("--- %s seconds ---" % (time.time() - start_time))