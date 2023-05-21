import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define the objective function to minimize
def objective_function(params):
    C = params[0]
    gamma = params[1]
    svm = SVC(C=C, gamma=gamma, kernel='rbf')
    svm.fit(X_train, y_train)
    accuracy = svm.score(X_test, y_test)
    return -accuracy  # maximize accuracy by minimizing negative accuracy

# Set the bounds for C and gamma
bounds = [(1e-10, 10000.0), (1e-10, 10000.0)]

# Define the PSO algorithm
class PSO:
    def __init__(self, num_particles, num_iterations, objective_function, bounds, inertia_weight=0.7, cognitive_weight=1.4, social_weight=1.4):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.objective_function = objective_function
        self.bounds = bounds
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.particles = []
        self.global_best_position = None
        self.global_best_value = None

    def optimize(self):
        # Initialize particles
        for i in range(self.num_particles):
            particle = Particle(self.bounds)
            self.particles.append(particle)

        # Run iterations
        for i in range(self.num_iterations):
            for particle in self.particles:
                # Evaluate particle's objective function value
                value = self.objective_function(particle.position)
                if particle.best_value is None or value < particle.best_value:
                    particle.best_position = particle.position.copy()
                    particle.best_value = value
                if self.global_best_value is None or value < self.global_best_value:
                    self.global_best_position = particle.position.copy()
                    self.global_best_value = value

            # Update particles' velocities and positions
            for particle in self.particles:
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)
                particle.velocity = self.inertia_weight * particle.velocity + self.cognitive_weight * r1 * (particle.best_position - particle.position) + self.social_weight * r2 * (self.global_best_position - particle.position)
                particle.position = np.clip(particle.position + particle.velocity, self.bounds[0], self.bounds[1])

# Define the Particle class for PSO
class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], size=2)
        self.velocity = np.zeros(2)
        self.best_position = None
        self.best_value = None

# Create a PSO optimizer with 10 particles
pso = PSO(num_particles=10, num_iterations=1000, objective_function=objective_function, bounds=bounds)

# Run the PSO optimization
pso.optimize()

# Train the final SVM model with the best parameters
C = pso.global_best_position[0]
gamma = pso.global_best_position[1]
svm = SVC(C=C, gamma=gamma, kernel='rbf')
svm.fit(X_train, y_train)
accuracy = svm.score(X_test, y_test)

# Print the best parameters and accuracy
print(f'Best parameters: C={C}, gamma={gamma}')
print(f'Accuracy: {accuracy}')