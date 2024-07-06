# %%
from node import MiniSom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
""" Test on the UCI seeds dataset """
columns = [
    "area",
    "perimeter",
    "compactness",
    "length_kernel",
    "width_kernel",
    "asymmetry_coefficient",
    "length_kernel_groove",
    "target",
]
data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
    names=columns,
    sep="\t+",
    engine="python",
)
target = data["target"].values
label_names = {1: "Kama", 2: "Rosa", 3: "Canadian"}
data = data[data.columns[:-1]]
# data normalization
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

# %% Initialization and training
n_neurons = 9
m_neurons = 9
som = MiniSom(
    n_neurons,
    m_neurons,
    data.shape[1],
    sigma=1.5,
    learning_rate=0.5,
    neighborhood_function="gaussian",
    random_seed=0,
    topology="rectangular",
)

som.pca_weights_init(data)
som.train(data, 1000, verbose=True)  # random training

# %%
"""To visualize the result of the training we can plot the distance map
(U-Matrix) using a pseudocolor where the neurons of the maps are displayed
as an array of cells and the color represents the (weights) distance from 
the neighbour neurons. On top of the pseudo color we can add markers that 
repesent the samples mapped in the specific cells."""

print("topographic error: ", som.topographic_error(data[:100]))
plt.figure(figsize=(9, 9))

plt.pcolor(
    som.distance_map().T, cmap="bone_r"
)  # plotting the distance map as background
plt.colorbar()

# Plotting the response for each pattern in the iris dataset
# different colors and markers for each label
markers = ["o", "s", "D"]
colors = ["C0", "C1", "C2"]
for cnt, xx in enumerate(data):
    w = som.winner(xx)  # getting the winner
    # palce a marker on the winning position for the sample xx
    plt.plot(
        w[0] + 0.5,
        w[1] + 0.5,
        markers[target[cnt] - 1],
        markerfacecolor="None",
        markeredgecolor=colors[target[cnt] - 1],
        markersize=12,
        markeredgewidth=2,
    )

plt.show()

# %%
"""To have an overview of how the samples are distributed across the map 
a scatter chart can be used where each dot represents the coordinates of
the winning neuron. A random offset is added to avoid overlaps between 
points within the same cell."""
w_x, w_y = zip(*[som.winner(d) for d in data])
w_x = np.array(w_x)
w_y = np.array(w_y)

plt.figure(figsize=(10, 9))
plt.pcolor(som.distance_map().T, cmap="bone_r", alpha=0.2)
plt.colorbar()

for c in np.unique(target):
    idx_target = target == c
    plt.scatter(
        w_x[idx_target] + 0.5 + (np.random.rand(np.sum(idx_target)) - 0.5) * 0.8,
        w_y[idx_target] + 0.5 + (np.random.rand(np.sum(idx_target)) - 0.5) * 0.8,
        s=50,
        c=colors[c - 1],
        label=label_names[c],
    )
plt.legend(loc="upper right")
plt.grid()
# plt.savefig('resulting_images/som_seed.png')
plt.show()

#%%
"""Test win_map functions, get_neighbor_data, and get_neighbor_indices functions"""
import jax.numpy as jnp

num_samples = 10
indices = jnp.arange(len(data))
winmap = som.win_map(data)
win_idx, win_data = som.win_map_index_n_data(indices, data)
candi_data, dists = som.get_neighbor_data(data[10], data, min_samples=num_samples)
candi_idx, dists2 = som.get_neighbor_indices(data[10], data, indices, min_samples=num_samples)  # this should throw a ValueError if given a relatively large num_samples

# %%
"""To have an idea of which neurons of the map are activated more often 
we can create another pseudocolor plot that reflects the activation 
frequencies"""
plt.figure(figsize=(7, 7))
frequencies = som.activation_response(data)
plt.pcolor(frequencies.T, cmap="Blues")
plt.colorbar()
plt.show()

# %%
"""When dealing with a supervised problem, one can visualize the proportion
of samples per class falling in a specific neuron using a pie chart per neuron"""
import matplotlib.gridspec as gridspec

labels_map = som.labels_map(data, [label_names[t] for t in target])

fig = plt.figure(figsize=(9, 9))
the_grid = gridspec.GridSpec(n_neurons, m_neurons, fig)
for position in labels_map.keys():
    label_fracs = [labels_map[position][l] for l in label_names.values()]
    plt.subplot(the_grid[n_neurons - 1 - position[1], position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)

plt.legend(patches, label_names.values(), bbox_to_anchor=(3.5, 6.5), ncol=3)
# plt.savefig('resulting_images/som_seed_pies.png')
plt.show()

# %%
"""To understand how the training evolves we can plot the quantization and
topographic error of the SOM at each step. This is particularly important 
when estimating the number of iterations to run"""
som = MiniSom(
    10,
    20,
    data.shape[1],
    sigma=3.0,
    learning_rate=0.7,
    neighborhood_function="gaussian",
    random_seed=10,
)

max_iter = 1000
q_error = []
t_error = []

for i in range(max_iter):
    rand_i = np.random.randint(len(data))
    som.update(data[rand_i], som.winner(data[rand_i]), i, max_iter)
    q_error.append(som.quantization_error(data))
    t_error.append(som.topographic_error(data))

plt.plot(np.arange(max_iter), q_error, label="quantization error")
plt.plot(np.arange(max_iter), t_error, label="topographic error")
plt.ylabel("error")
plt.xlabel("iteration index")
plt.legend()
plt.show()

# %%
# with PCA initialization
som = MiniSom(
    10,
    20,
    data.shape[1],
    sigma=3.0,
    learning_rate=0.7,
    neighborhood_function="gaussian",
    random_seed=10,
)

som.pca_weights_init(data)
max_iter = 1000
q_error = []
t_error = []

for i in range(max_iter):
    rand_i = np.random.randint(len(data))
    som.update(data[rand_i], som.winner(data[rand_i]), i, max_iter)
    q_error.append(som.quantization_error(data))
    t_error.append(som.topographic_error(data))

plt.plot(np.arange(max_iter), q_error, label="quantization error")
plt.plot(np.arange(max_iter), t_error, label="topographic error")
plt.ylabel("error")
plt.xlabel("iteration index")
plt.legend()
plt.show()

# %%
"""Test on the UCI hand-written digit (8x8) dataset"""
from sklearn import datasets
from sklearn.preprocessing import scale

# load the digits dataset from scikit-learn
digits = datasets.load_digits(n_class=10)
data = digits.data  # matrix where each row is a vector that represent a digit.
data = scale(data)
num = digits.target  # num[i] is the digit represented by data[i]

som = MiniSom(
    30,
    30,
    64,
    sigma=4,
    random_seed=10,
    learning_rate=0.5,
    neighborhood_function="triangle",
)
som.pca_weights_init(data)
som.train(data, 5000, random_order=True, verbose=True)  # random training

# %%
"""Each input vector for the SOM represents the entire image obtained 
reshaping the original image of dimension 8-by-8 into a vector of 64 elements. 
The images in input are gray scale."""
# place each digit on the map represented by the SOM
plt.figure(figsize=(8, 8))
wmap = {}
im = 0
for x, t in zip(data, num):  # scatterplot
    w = som.winner(x)
    w = tuple(np.asarray(w))
    wmap[w] = im
    plt.text(
        w[0] + 0.5,
        w[1] + 0.5,
        str(t),
        color=plt.cm.rainbow(t / 10.0),
        fontdict={"weight": "bold", "size": 11},
    )
    im = im + 1
plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])
# plt.savefig('resulting_images/som_digts.png')
plt.show()

plt.figure(figsize=(10, 10), facecolor="white")
cnt = 0
for j in reversed(range(20)):  # images mosaic
    for i in range(20):
        plt.subplot(20, 20, cnt + 1, frameon=False, xticks=[], yticks=[])
        if (i, j) in wmap:
            plt.imshow(
                digits.images[wmap[(i, j)]], cmap="Greys", interpolation="nearest"
            )
        else:
            plt.imshow(np.zeros((8, 8)), cmap="Greys")
        cnt = cnt + 1

plt.tight_layout()
# plt.savefig('resulting_images/som_digts_imgs.png')
plt.show()
