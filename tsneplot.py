import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def read_weights(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    weights = []
    for line in lines:
        if not ("Epoch" in line):
            line = line.strip() 
            values = line.split()
            weight_row = [float(value) for value in values]  
            weights.append(weight_row) 
    
    return np.array(weights)  

trajectories = []
for i in range(1, 16):
    filename = f"weights_run_{i}.txt"
    trajectories.append(read_weights(filename))

all_weights = np.vstack(trajectories)

tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(all_weights)


plt.figure(figsize=(10, 8))

start = 0
for i, trajectory in enumerate(trajectories):
    length = len(trajectory)
    reduced_trajectory = reduced[start:start + length]
    start += length

    if i+1 <= 5:
        pass
        # plt.plot(reduced_trajectory[:, 0], reduced_trajectory[:, 1], label=f"GD Run {i + 1}" , marker="o", markersize="3")
    elif i+1 <= 10:
        pass
        # plt.plot(reduced_trajectory[:, 0], reduced_trajectory[:, 1], label=f"SGD Run {i + 1 - 5}", marker="d", markersize="2")
    elif i+1 <= 15:
        plt.plot(reduced_trajectory[:, 0], reduced_trajectory[:, 1], label=f"ADAM Run {i + 1 -10}", marker="s", markersize="1")
    

plt.title("t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.show()
