# %% Import packages
from prisoner.agent import Agent, AgentNoMemory, SimpleAgent
from prisoner.environment import PrisonersDilemma
from prisoner.genetic_algorithm import GeneticAlgorithm, load

import torch
import pickle
import datetime
import argparse

time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# %% Set Parameters with argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--n-generations", type=int, default=100, help="Number of generations"
)
parser.add_argument("--n-rounds", type=int, default=100, help="Number of rounds")
parser.add_argument(
    "--rounds-var", type=float, default=0.0, help="Variance of rounds"
)
parser.add_argument("--n-agents", type=int, default=100, help="Number of agents")
parser.add_argument("--mutation-rate", type=float, default=0.05, help="Mutation rate")
parser.add_argument("--num-elites", type=int, default=20, help="Number of elites")
parser.add_argument(
    "--num-random-saved", type=int, default=10, help="Number of random saved"
)
parser.add_argument("--use-cuda", action="store_true", help="Use CUDA")
parser.add_argument("--agent", type=str, default="memory", help="Agent class", choices=["memory", "no-memory", "simple"])
parser.add_argument("--seed", type=int, default=0, help="Random seed")
args = parser.parse_args()

args.use_cuda = args.use_cuda and torch.cuda.is_available()
# torch set global default device to cuda if available
torch.set_default_tensor_type(
    torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
)
torch.manual_seed(args.seed)

# %% initialize game and genetic algorithm
game = PrisonersDilemma(num_rounds=args.n_rounds, rounds_var=args.rounds_var)
if args.agent == "memory":
    agent_class = Agent
elif args.agent == "no-memory":
    agent_class = AgentNoMemory
elif args.agent == "simple":
    agent_class = SimpleAgent
genetic_algorithm = GeneticAlgorithm(
    game,
    agent_class=agent_class,
    population_size=args.n_agents,
    mutation_rate=args.mutation_rate,
    num_elites=args.num_elites,
    num_random_saved=args.num_random_saved,
)

# %% run genetic algorithm
if __name__ == "__main__":
    print(args)
    history = genetic_algorithm.run(
        args.n_generations,
        show_progress=True,
        save_history=f"files/history_{time}.pickle",
    )
    genetic_algorithm.save(f"files/genetic_algorithm_{time}.pickle")

# %% load genetic algorithm
# genetic_algorithm = load("files/genetic_algorithm_20231227172018.pickle")
