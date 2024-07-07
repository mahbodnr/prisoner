from prisoner.agent import Agent, AgentNoMemory, SimpleAgent
from prisoner.environment import Game

import torch
import tqdm
import pickle

class GeneticAlgorithm:
    def __init__(self,
            game: Game,
            agent_class=Agent,
            population_size=100,
            mutation_rate=0.01,
            num_elites=10,
            num_random_saved=0,
        ):
        self.game = game
        self.agent_class = agent_class
        self.population_size = population_size
        self.parameter_mutation_rate = mutation_rate
        self.num_elites = num_elites
        self.num_random_saved = num_random_saved

        self.reset()

    def initialize_population(self):
        self.population = []
        for i in range(self.population_size):
            if self.agent_class == Agent:
                agent_weights = {
                    "rnn_weights_ih": torch.randn(Agent.hidden_size, Agent.input_size),
                    "rnn_weights_hh": torch.randn(Agent.hidden_size, Agent.hidden_size),
                    "rnn_bias_ih": torch.randn(Agent.hidden_size),
                    "rnn_bias_hh": torch.randn(Agent.hidden_size),
                    "output_weights": torch.randn(Agent.output_size, Agent.hidden_size),
                    "output_bias": torch.randn(Agent.output_size)
                }
            elif self.agent_class == AgentNoMemory:
                agent_weights = {
                    "output_weights": torch.randn(AgentNoMemory.output_size, AgentNoMemory.input_size),
                    "output_bias": torch.randn(AgentNoMemory.output_size)
                }
            elif self.agent_class == SimpleAgent:
                agent_weights = {
                    "output_weights": torch.randn(SimpleAgent.output_size, SimpleAgent.input_size),
                    "output_bias": torch.randn(SimpleAgent.output_size)
                }
            else:
                raise ValueError(f"Unknown agent class {self.agent_class}")
            agent = {
                "weights": agent_weights,
                "id": next(self._get_id),
                "parent1": None,
                "parent2": None,
            }
            self.population.append(agent)

    def reset(self):
        self._get_id = self.id_generator(0)
        self.initialize_population()
        self.history = {
            "game_scores": [],
            "population": [self.deep_copy_population()]
        }

    def run_generation(self):
        fitness_scores, game_scores = self.get_fitness_scores() # TODO: add num repeats for games
        self.history["game_scores"].append(game_scores.clone())
        saved = self.get_saved(fitness_scores)
        new_population = self.get_offspring(saved, fitness_scores)
        self.population = saved + new_population
        self.history["population"].append(self.deep_copy_population())

    def get_saved(self, fitness_scores):
        _, indices = torch.sort(fitness_scores, descending=True)
        saved = []
        # add elites
        for i in range(self.num_elites):
            saved.append(self.population[indices[i]])
        # add random agents from the rest of the population
        random_indices = torch.randperm(self.population_size - self.num_elites)[:self.num_random_saved]
        for i in random_indices:
            saved.append(self.population[i])
        return saved

    def get_fitness_scores(self):
        game_scores = torch.zeros(self.population_size, self.population_size)
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                agent1 = self.agent_class(**self.population[i]["weights"])
                agent2 = self.agent_class(**self.population[j]["weights"])
                agent1_score, agent2_score = self.game.run(agent1, agent2)
                game_scores[i][j] = agent1_score
                game_scores[j][i] = agent2_score

        fitness_scores = torch.sum(game_scores, dim=1)
        return fitness_scores, game_scores

    def get_offspring(self, elites, fitness_scores):
        fitness_scores = fitness_scores / torch.sum(fitness_scores)
        new_population = []
        for i in range(self.population_size - len(elites)):
            parent1, parent2 = self.select_parents(fitness_scores)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        return new_population

    def select_parents(self, fitness_scores):
        parent1_index = torch.multinomial(fitness_scores, 1).item()
        parent2_index = torch.multinomial(fitness_scores, 1).item()
        return self.population[parent1_index], self.population[parent2_index]

    def crossover(self, parent1, parent2):
        child = {
            "weights": {},
            "parent1": parent1["id"],
            "parent2": parent2["id"],
            "id": next(self._get_id),
        }
        # TODO: add more crossover methods (e.g. Average, BLX-alpha, Simulated Binary Crossover, etc.)
        for key in parent1["weights"]:
            if torch.rand(1).item() < 0.5:
                child["weights"][key] = parent1["weights"][key].clone()
            else:
                child["weights"][key] = parent2["weights"][key].clone()
        return child

    def mutate(self, child):
        for key, value in child["weights"].items():
            mutated_indices = torch.rand_like(value) < self.parameter_mutation_rate
            child["weights"][key][mutated_indices] = torch.randn_like(value)[mutated_indices]
        return child

    def deep_copy_population(self):
        return [self._deep_copy_agent(agent) for agent in self.population]

    def _deep_copy_agent(self, agent):
        return {
            "weights": {key: value.clone() for key, value in agent["weights"].items()},
            "id": agent["id"],
            "parent1": agent["parent1"],
            "parent2": agent["parent2"],
        }

    def run(self, num_generations, show_progress=False, save_history=None):
        if show_progress:
            p = tqdm.tqdm(range(num_generations))
        else:
            p = range(num_generations)
        for _ in p:
            self.run_generation()
            if save_history:
                with open(save_history, "wb") as f:
                    pickle.dump(self.history, f)
        return self.history

    def id_generator(self, start=0):
        i = start
        while True:
            yield i
            i += 1

    def save(self, filename):
        state = {
            "population": self.population,
            "history": self.history,
            "game": self.game,
            "mutation_rate": self.parameter_mutation_rate,
            "num_elites": self.num_elites,
            "num_random_saved": self.num_random_saved,
            "start_id": next(self._get_id),
        }
        with open(filename, "wb") as f:
            pickle.dump(state, f)

def load(filename):
    with open(filename, "rb") as f:
        state = pickle.load(f)
    genetic_algorithm = GeneticAlgorithm(
        state["game"],
        population_size=len(state["population"]),
        mutation_rate=state["mutation_rate"],
        num_elites=state["num_elites"],
        num_random_saved=state["num_random_saved"],
    )
    genetic_algorithm.population = state["population"]
    genetic_algorithm.history = state["history"]
    genetic_algorithm._get_id = genetic_algorithm.id_generator(state["start_id"])
    return genetic_algorithm