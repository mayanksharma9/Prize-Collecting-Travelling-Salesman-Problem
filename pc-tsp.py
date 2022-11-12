import math
import random
import numpy as np;


class City:
    def __init__(self, name, lat, long, prize):
        self.name = name
        self.lat = lat
        self.long = long
        self.prize = prize

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.name} @ ({self.lat}, {self.long}) | {self.prize}"


def distance(l1: City, l2: City):
    return 3963.0 * math.acos(
        min(
            1,
            (math.sin(math.radians(l1.lat)) * math.sin(math.radians(l2.lat)))
            + math.cos(math.radians(l1.lat))
            * math.cos(math.radians(l2.lat))
            * math.cos(math.radians(l2.long) - math.radians(l1.long)),
        )
    )


locations = []
f = open("D:\CSUDH\First Semester\Computer Architechture-531\Travelling salesman problem\cities.txt")
for line in f:
    name, lat, long, prize = line.split("|")
    locations.append(City(name, np.double(lat), np.double(long), np.double(prize)))

n_dest = len(locations)
dist_mat = np.zeros([n_dest, n_dest])
for i, loc in enumerate(locations):
    for j, loc2 in enumerate(locations):
        d = 0.1 * distance(loc, loc2)
        dist_mat[i, j] = d


def update_q(q, dist_mat, reward, state, action, alpha=0.01, gamma=0.3):
    immediate_reward = reward - dist_mat[state, action]
    delayed_reward = q[action, :].max()
    q[state, action] += alpha * (reward + gamma * delayed_reward - q[state, action])
    return q


reward_to_collect = 100
q = np.zeros([n_dest, n_dest])
epsilon = 1.0
n_train = 20000
for i in range(n_train):
    reward_collected = 0
    history = [0]
    state = 0
    remainaing_cities = [dest for dest in range(n_dest) if dest not in history]
    while remainaing_cities:
        if random.random() < epsilon:
            action = random.choice(remainaing_cities)
        else:
            best_action_index = q[state, remainaing_cities].argmax()
            action = remainaing_cities[best_action_index]

        reward_collected = (
            reward_collected + locations[action].prize - dist_mat[state, action]
        )

        q = update_q(q, dist_mat, reward_collected, state, action)
        history.append(action)
        state = action
        remainaing_cities.remove(action)
        if (
            reward_collected
            + (locations[history[0]].prize - dist_mat[state, history[0]])
            - sum(locations[i].prize for i in remainaing_cities)
            >= reward_to_collect
        ):
            break

    action = 0
    reward_collected = (
        reward_collected + locations[action].prize - dist_mat[state, action]
    )
    q = update_q(q, dist_mat, reward_collected, state, action)
    history.append(0)
    epsilon = 1.0 - i * 1 / n_train


history = [0]
state = 0
remainaing_cities = [dest for dest in range(n_dest) if dest not in history]
reward_collected = 0
while remainaing_cities:
    best_action_index = q[state, remainaing_cities].argmax()
    action = remainaing_cities[best_action_index]
    history.append(action)
    reward_collected = (
        reward_collected + locations[action].prize - dist_mat[state, action]
    )
    state = action
    remainaing_cities.remove(action)
    if (
        reward_collected
        + (locations[history[0]].prize - dist_mat[state, history[0]])
        - sum(locations[i].prize for i in remainaing_cities)
        >= reward_to_collect
    ):
        break

action = history[0]
history.append(action)
reward_collected += locations[action].prize - dist_mat[state, action]

print("Travel Path:")
print(" -> ".join([str(b) for b in history]))
print(f"Reward collected: {reward_collected}")
