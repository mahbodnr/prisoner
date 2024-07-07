from prisoner.agent import Agent

import torch


class Game:
    pass


class PrisonersDilemma(Game):
    def __init__(
        self,
        reward=3,
        temptation=5,
        punishment=1,
        sucker=0,
        num_rounds=10,
        rounds_var=0,
        randomness=0,
        record_history=False,
        min_rounds=1,
    ):
        self.reward = reward
        self.temptation = temptation
        self.punishment = punishment
        self.sucker = sucker

        self.num_rounds_mean = num_rounds
        self.rounds_var = rounds_var
        self.randomness = randomness
        assert self.randomness == 0, "Randomness not yet implemented"
        self.record_history = record_history
        self.min_rounds = min_rounds
        assert self.min_rounds >= 0, "min_rounds must be non-negative"

    def run(self, agent1: Agent, agent2: Agent):
        if self.record_history:
            self.history = {
                "agent1_actions": [],
                "agent2_actions": [],
                "agent1_scores": [],
                "agent2_scores": [],
            }
        agent1_score = 0
        agent2_score = 0
        agent1_hidden = None
        agent2_hidden = None
        agent1_input = torch.zeros(1, 2)
        agent2_input = torch.zeros(1, 2)

        for _ in range(self.num_rounds()):
            # TODO: save hidden inside the agent
            agent1_action, agent1_hidden = agent1(agent1_input, agent1_hidden)
            agent2_action, agent2_hidden = agent2(agent2_input, agent2_hidden)
            agent1_action_score, agent2_action_score = self.get_action_scores(
                agent1_action.item(), agent2_action.item()
            )
            agent1_score += agent1_action_score
            agent2_score += agent2_action_score
            agent1_input = torch.tensor([[agent1_action, agent2_action]])
            agent2_input = torch.tensor([[agent2_action, agent1_action]])

            if self.record_history:
                self.history["agent1_actions"].append(agent1_action)
                self.history["agent2_actions"].append(agent2_action)
                self.history["agent1_scores"].append(agent1_score)
                self.history["agent2_scores"].append(agent2_score)

        return agent1_score, agent2_score

    def num_rounds(self):
        return max(
            int(torch.normal(self.num_rounds_mean, self.rounds_var, (1,)).item()),
            self.min_rounds,
        )

    def get_action_scores(self, agent1_action, agent2_action):
        if agent1_action == 1 and agent2_action == 1:
            return self.reward, self.reward
        elif agent1_action ==1  and agent2_action == -1:
            return self.sucker, self.temptation
        elif agent1_action == -1 and agent2_action == 1:
            return self.temptation, self.sucker
        elif agent1_action == -1 and agent2_action == -1:
            return self.punishment, self.punishment
        else:
            raise ValueError(f"Invalid action pair: {agent1_action}, {agent2_action}")

    def play_with_agent(self, agent):
        print("Playing with agent. Select action: 1 or -1")
        action = input("Enter action: ")
        agent_input = torch.zeros(1, 2)
        agent_hidden = None

        while action != "q":
            action = int(action)
            assert action in [-1, 1], "Action must be -1 or 1"
            agent_action, agent_hidden = agent(agent_input, agent_hidden)
            agent_input = torch.tensor([[agent_action, action]])
            print(f"Agent action: {int(agent_action.item())}")
            action = input("Enter action: ")
