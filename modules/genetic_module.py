import streamlit as st
import random

def run_genetic():
    st.header("🧬 Genetic Algorithm")

    GENES_LENGTH = st.number_input("Gene Length", min_value=2, value=6, step=1, key="genetic_length")
    POPULATION_SIZE = st.number_input("Population Size", min_value=2, value=6, step=1, key="genetic_pop")
    GENERATIONS = st.number_input("Generations", min_value=1, value=3, step=1, key="genetic_gen")

    if st.button("Run Genetic Algorithm"):
        genetic_algorithm(GENES_LENGTH, POPULATION_SIZE, GENERATIONS)

def fitness(individual):
    return sum(individual)

def generate_individual(length):
    return [random.randint(0, 1) for _ in range(length)]

def generate_population(size, length):
    return [generate_individual(length) for _ in range(size)]

def selection(population):
    return sorted(population, key=fitness, reverse=True)[:2]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    return (
        parent1[:point] + parent2[point:],
        parent2[:point] + parent1[point:]
    )

def mutate(individual, mutation_rate=0.2):
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            old = mutated[i]
            mutated[i] = 1 - mutated[i]
            st.text(f"  Mutation at gene {i}: {old} → {mutated[i]}")
    return mutated

def genetic_algorithm(length, population_size, generations):
    population = generate_population(population_size, length)

    for generation in range(generations):
        st.subheader(f"🔁 Generation {generation + 1}")
        for i, ind in enumerate(population):
            st.text(f"Individual {i + 1}: {ind} - Fitness: {fitness(ind)}")

        parent1, parent2 = selection(population)
        st.text(f"\nSelected Parents:\n Parent 1: {parent1} - Fitness: {fitness(parent1)}\n Parent 2: {parent2} - Fitness: {fitness(parent2)}")

        child1, child2 = crossover(parent1, parent2)
        st.text(f"Child 1 (before mutation): {child1}")
        st.text(f"Child 2 (before mutation): {child2}")

        child1 = mutate(child1)
        child2 = mutate(child2)

        st.text(f"Child 1 (after mutation):  {child1}")
        st.text(f"Child 2 (after mutation):  {child2}")

        new_population = []
        while len(new_population) < population_size:
            new_population.append(child1.copy())
            if len(new_population) < population_size:
                new_population.append(child2.copy())

        population = new_population

    best = max(population, key=fitness)
    st.success(f"🏆 Best solution: {best} - Fitness: {fitness(best)}")
