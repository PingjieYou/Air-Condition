import matplotlib.pyplot as plt
import csv
import pandas as pd


fitness_values = []
infeas = []
turns =[]
length = []
filename = 'data.csv'
with open(filename,'r') as csvfile:
        lines = csv.DictReader(csvfile,fieldnames=['Path Length', 'Turns', 'Infeasible Steps', 'Fitness'])
        next(lines)
        for r in lines:
            fitness_values.append(float(r['Fitness']))
            length.append(float(r['Path Length']))
            turns.append(float(r['Turns']))
            infeas.append(float(r['Infeasible Steps']))
x = list(range(1,len(fitness_values)+1))



# Convert the data to a pandas DataFrame
df = pd.DataFrame({'Infeasible Steps': infeas, 'Path Length': length, 'Turns': turns, 'Fitness': fitness_values})

# Apply a moving average with a window size of 5
smoothed_df = df.rolling(window=5).mean()

# Plot the smoothed data
plt.figure(figsize=(6, 8))

plt.subplot(4, 1, 1)
plt.plot(x, smoothed_df['Infeasible Steps'])
plt.ylabel('Infeasible Steps')

plt.subplot(4, 1, 2)
plt.plot(x, smoothed_df['Path Length'])
plt.ylabel('Path Length')

plt.subplot(4, 1, 3)
plt.plot(x, smoothed_df['Turns'])
plt.ylabel('Turns')

plt.subplot(4, 1, 4)
plt.plot(x, smoothed_df['Fitness'])
plt.xlabel('Generations')
plt.ylabel('Fitness')

plt.show()