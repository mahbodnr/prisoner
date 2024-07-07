# %%
import pickle
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from prisoner.agent import Agent, AgentNoMemory, SimpleAgent
from prisoner.environment import PrisonersDilemma
# %%
filename = "files/history_20231230185017.pickle"
with open(filename, "rb") as f:
    history = pickle.load(f)
population = history["population"]
game_scores = history["game_scores"]
fitness_scores = list(map(lambda x: x.sum().item(), game_scores))
# %%
population
# %%
plt.plot(fitness_scores)
plt.show()
for i, gs in enumerate(game_scores):
    if i % 20 == 0:
        print(i)
        plt.imshow(gs.detach().cpu().numpy())
        # print(list(map(lambda x: x["id"], population[i]))[:20])
        plt.show()
# %% SimpleAgent
givers = []
takers = []
tft = []
confused = []

for p in population:
    g = 0
    t = 0
    tf = 0
    c = 0
    input = torch.tensor([[0,-1], [0,1]], dtype=torch.float)
    for a in p:
        agent = SimpleAgent(**a["weights"])
        output = agent(input)[0]
        if output[0][0] == output[1][0]== 1:
            g += 1
        elif output[0][0] == output[1][0]== -1:
            t += 1
        elif output[0][0] == -1 and output[1][0] == 1:
            tf += 1
        elif output[0][0] == 1 and output[1][0] == -1:
            c += 1
        else:
            raise ValueError(f"Invalid output: {output}")
        
    givers.append(g)
    takers.append(t)
    tft.append(tf)
    confused.append(c)

plt.plot(givers, label="givers")
plt.plot(takers, label="takers")
plt.plot(tft, label="tft")
plt.plot(confused, label="confused")
plt.legend()
plt.show()
# %%
for i, p in enumerate(population):
    if i % 20 == 0:
        weights = []
        biases = []
        for agent in p:
            weights.append(agent["weights"]["output_weights"].item())
            biases.append(agent["weights"]["output_bias"].item())
        # add x=0 and y=0 lines
        plt.plot([0,0], [min(biases), max(biases)], "k--")
        plt.plot([min(weights), max(weights)], [0,0], "k--")
        plt.plot(weights, biases, "o")
        plt.show()
# %%
# for p in population:
#     print(sorted(list(map(lambda x: x["id"], p))))

# %%
last_population = population[-1]
last_fitness = game_scores[-1].sum(1)
best_agent_state = last_population[torch.argmax(last_fitness)]
# %%

plt.subplot(2,3,1)
plt.imshow(best_agent_state["weights"]['rnn_weights_ih'])
plt.subplot(2,3,2)
plt.imshow(best_agent_state["weights"]['rnn_weights_hh'])
plt.subplot(2,3,3)
plt.imshow(best_agent_state["weights"]['output_weights'])
plt.subplot(2,3,4)
plt.imshow(best_agent_state["weights"]['rnn_bias_ih'].view(-1,1))
plt.subplot(2,3,5)
plt.imshow(best_agent_state["weights"]['rnn_bias_hh'].view(-1,1))
plt.subplot(2,3,6)
plt.imshow(best_agent_state["weights"]['output_bias'].view(-1,1))
plt.show()
# %%
plt.subplot(1,2,1)
plt.imshow(best_agent_state["weights"]['output_weights'])
plt.subplot(1,2,2)
plt.imshow(best_agent_state["weights"]['output_bias'].view(-1,1))
plt.colorbar()
plt.show()
agent = AgentNoMemory(**best_agent_state["weights"])
input = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float)
sns.heatmap(agent(input)[0].reshape(2,2).detach().numpy(), annot=True, fmt=".0f", cbar=False)
# %%
agent = Agent(**best_agent_state["weights"])
game = PrisonersDilemma(num_rounds=100, rounds_var=0)
game.play_with_agent(agent)



# %%
