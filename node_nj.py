#%%
import jax
from jax import random, numpy as jnp

from collections import defaultdict, Counter
from warnings import warn
from sys import stdout
from time import time
from datetime import timedelta
import pickle
import os
import numpy as np
import ninjax as nj

"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
"""


def _build_iteration_indexes(
    data_len, num_iterations, verbose=False, random_key=None, use_epochs=False
):
    """Returns an iterable with the indexes of the samples
    to pick at each iteration of the training.

    If random_generator is not None, it must be an instance
    of numpy.random.RandomState and it will be used
    to randomize the order of the samples."""
    if use_epochs:
        iterations_per_epoch = jnp.arange(data_len)
        if random_key is not None:
            iterations_per_epoch = jax.random.permutation(
                random_key, iterations_per_epoch
            )
        iterations = jnp.tile(iterations_per_epoch, num_iterations)
    else:
        iterations = jnp.arange(num_iterations) % data_len
        # iterations = jnp.arange(data_len)
        if random_key is not None:
            iterations = jax.random.permutation(random_key, iterations)
    if verbose:
        return _wrap_index__in_verbose(iterations)
    else:
        return iterations


def _wrap_index__in_verbose(iterations):
    """Yields the values in iterations printing the status on the stdout."""
    m = len(iterations)
    digits = len(str(m))
    progress = "\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s"
    progress = progress.format(m=m, d=digits, s=0)
    stdout.write(progress)
    beginning = time()
    stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m - i + 1) * (time() - beginning)) / (i + 1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = "\r [ {i:{d}} / {m} ]".format(i=i + 1, d=digits, m=m)
        progress += " {p:3.0f}%".format(p=100 * (i + 1) / m)
        progress += " - {time_left} left ".format(time_left=time_left)
        stdout.write(progress)


def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array."""
    return jnp.sqrt(jnp.dot(x, x.T))


def asymptotic_decay(learning_rate: float, t: int, max_iter: int):
    """Decay function of the learning process.
    Parameters
    ----------
    learning_rate : float
        current learning rate.

    t : int
        current iteration.

    max_iter : int
        maximum number of iterations for the training.
    """
    return learning_rate / (1 + t / (max_iter / 2))


class MiniSom(nj.Module):
    Y_HEX_CONV_FACTOR = (3.0 / 2.0) / jnp.sqrt(3)

    def __init__(
        self,
        x,
        y,
        input_len,
        sigma=1.0,
        learning_rate=0.5,
        decay_function=asymptotic_decay,
        neighborhood_function="gaussian",
        topology="rectangular",
        activation_distance="euclidean",
        verbose=True,
        use_epochs=False,
    ):
        """Initializes a Self Organizing Maps.

        A rule of thumb to set the size of the grid for a dimensionality
        reduction task is that it should contain 5*sqrt(N) neurons
        where N is the number of samples in the dataset to analyze.

        E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
        hence a map 8-by-8 should perform well.

        Parameters
        ----------
        x : int
            x dimension of the SOM.

        y : int
            y dimension of the SOM.

        input_len : int
            Number of the elements of the vectors in input.

        sigma : float, optional (default=1.0)
            Spread of the neighborhood function, needs to be adequate
            to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T)
            where T is #num_iteration/2)
        learning_rate : initial learning rate
            (at the iteration t we have
            learning_rate(t) = learning_rate / (1 + t/T)
            where T is #num_iteration/2)

        decay_function : function (default=asymptotic_decay)
            Function that reduces learning_rate and sigma at each iteration
            the default function is:
                        learning_rate / (1+t/(max_iterarations/2))

            A custom decay function will need to to take in input
            three parameters in the following order:

            1. learning rate
            2. current iteration
            3. maximum number of iterations allowed


            Note that if a lambda function is used to define the decay
            MiniSom will not be pickable anymore.

        neighborhood_function : string, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map.
            Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'

        topology : string, optional (default='rectangular')
            Topology of the map.
            Possible values: 'rectangular', 'hexagonal'

        activation_distance : string, callable optional (default='euclidean')
            Distance used to activate the map.
            Possible values: 'euclidean', 'cosine', 'manhattan', 'chebyshev'

            Example of callable that can be passed:

            def euclidean(x, w):
                return linalg.norm(subtract(x, w), axis=-1)

         verbose : bool (default=False)
            If True the status of the training will be
            printed each time the weights are updated.

        use_epochs : bool (default=False)
            If True the SOM will be trained for num_iteration epochs.
            In one epoch the weights are updated len(data) times and
            the learning rate is constat throughout a single epoch.
        """
        if sigma >= x or sigma >= y:
            warn("Warning: sigma is too high for the dimension of the map.")

        self.x = x
        self.y = y
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        self.verbose = verbose
        self.use_epochs = use_epochs

        if topology not in ["hexagonal", "rectangular"]:
            msg = "%s not supported only hexagonal and rectangular available"
            raise ValueError(msg % topology)
        self.topology = topology
        
        self.neighborhood_function =neighborhood_function
        self._decay_function = decay_function
        # winner map
        self.winmap = defaultdict(list)

        neig_functions = {
            "gaussian": self._gaussian,
            "mexican_hat": self._mexican_hat,
            "bubble": self._bubble,
            "triangle": self._triangle,
        }

        if neighborhood_function not in neig_functions:
            msg = "%s not supported. Functions available: %s"
            raise ValueError(
                msg % (neighborhood_function, ", ".join(neig_functions.keys()))
            )

        if neighborhood_function in ["triangle", "bubble"] and (
            jnp.divmod(sigma, 1)[1] != 0 or sigma < 1
        ):
            warn(
                "sigma should be an integer >=1 when triangle or bubble"
                + "are used as neighborhood function"
            )

        self.neighborhood = neig_functions[neighborhood_function]

        distance_functions = {
            "euclidean": self._euclidean_distance,
            "cosine": self._cosine_distance,
            "manhattan": self._manhattan_distance,
            "chebyshev": self._chebyshev_distance,
        }

        if isinstance(activation_distance, str):
            if activation_distance not in distance_functions:
                msg = "%s not supported. Distances available: %s"
                raise ValueError(
                    msg % (activation_distance, ", ".join(distance_functions.keys()))
                )

            self._activation_distance = distance_functions[activation_distance]
        elif callable(activation_distance):
            self._activation_distance = activation_distance
    
    def init_weights(self, key):
        # random initialization
        key, new_key = random.split(key)
        self.get('_weights', random.uniform, key, (self.x, self.y, self._input_len), minval=-1, maxval=1)
        self.put('_weights', self.get('_weights') / jnp.linalg.norm(self.get('_weights'), axis=-1, keepdims=True))
        self.get('_activation_map', jnp.zeros, (self.x, self.y))
        self.get('_neigx', jnp.arange, self.x)
        self.get('_neigy', jnp.arange, self.y)  # used to evaluate the neighborhood function
        self._xx, self._yy = jnp.meshgrid(self.get('_neigx'), self.get('_neigy'))
        self._xx = self._xx.astype(jnp.float32)
        self._yy = self._yy.astype(jnp.float32)
        self.put('_xx', self._xx)
        self.put('_yy', self._yy)
        if self.topology == "hexagonal":
            self._xx = self._xx.at[::-2].add(-0.5)
            self._yy *= self.Y_HEX_CONV_FACTOR
            self.put('_xx', self._xx)
            self.put('_yy', self._yy)
            if self.neighborhood_function in ["triangle"]:
                warn(
                    "triangle neighborhood function does not "
                    + "take in account hexagonal topology"
                )
        

    def get_weights(self):
        """Returns the weights of the neural network."""
        return self.get('_weights')

    def get_euclidean_coordinates(self):
        """Returns the position of the neurons on an euclidean
        plane that reflects the chosen topology in two meshgrids xx and yy.
        Neuron with map coordinates (1, 4) has coordinate (xx[1, 4], yy[1, 4])
        in the euclidean plane.

        Only useful if the topology chosen is not rectangular.
        """
        return self.get('_xx').T, self.get('_yy').T

    def convert_map_to_euclidean(self, xy):
        """Converts map coordinates into euclidean coordinates
        that reflects the chosen topology.

        Only useful if the topology chosen is not rectangular.
        """
        return self.get('_xx').T[xy], self.get('_yy').T[xy]

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
        the element i,j is the response of the neuron i,j to x."""
        self.put('_activation_map', self._activation_distance(x, self.get('_weights')))

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self.get('_activation_map')

    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2 * sigma * sigma
        ax = jnp.exp(-jnp.power(self.get('_xx') - self.get('_xx').T[c], 2) / d)
        ay = jnp.exp(-jnp.power(self.get('_yy') - self.get('_yy').T[c], 2) / d)
        return (ax * ay).T  # the external product gives a matrix

    def _mexican_hat(self, c, sigma):
        """Mexican hat centered in c."""
        p = jnp.power(self.get('_xx') - self.get('_xx').T[c], 2) + jnp.power(
            self.get('_yy') - self.get('_yy').T[c], 2
        )
        d = 2 * sigma * sigma
        return (jnp.exp(-p / d) * (1 - 2 / d * p)).T

    def _bubble(self, c, sigma):
        """Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        ax = jnp.logical_and(self.get('_neigx') > c[0] - sigma, self.get('_neigx') < c[0] + sigma)
        ay = jnp.logical_and(self.get('_neigy') > c[1] - sigma, self.get('_neigy') < c[1] + sigma)
        return jnp.outer(ax, ay) * 1.0

    def _triangle(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        triangle_x = (-jnp.abs(c[0] - self.get('_neigx'))) + sigma
        triangle_y = (-jnp.abs(c[1] - self.get('_neigy'))) + sigma
        triangle_x = jnp.where(triangle_x < 0, 0., triangle_x)
        triangle_y = jnp.where(triangle_y < 0, 0., triangle_y)
        return jnp.outer(triangle_x, triangle_y)

    def _cosine_distance(self, x, w):
        num = (w * x).sum(axis=2)
        denum = jnp.multiply(jnp.linalg.norm(w, axis=2), jnp.linalg.norm(x))
        return 1 - num / (denum + 1e-8)

    def _euclidean_distance(self, x, w):
        x = jnp.asarray(x)
        return jnp.linalg.norm(jnp.subtract(x, w), axis=-1)

    def _manhattan_distance(self, x, w):
        x = jnp.asarray(x)
        return jnp.linalg.norm(jnp.subtract(x, w), ord=1, axis=-1)

    def _chebyshev_distance(self, x, w):
        x = jnp.asarray(x)
        return jnp.max(jnp.subtract(x, w), axis=-1)

    def _check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError("num_iteration must be > 1")

    def _check_input_len(self, data):
        """Checks that the data in input is of the correct shape."""
        data_len = len(data[0])
        if self._input_len != data_len:
            msg = "Received %d features, expected %d." % (data_len, self._input_len)
            raise ValueError(msg)

    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        coord = jnp.unravel_index(
            self.get('_activation_map').argmin(), self.get('_activation_map').shape
        )
        return coord

    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.

        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            rate of decay for sigma and learning rate
        max_iteration : int
            If use_epochs is True:
                Number of epochs the SOM will be trained for
            If use_epochs is False:
                Maximum number of iterations (one iteration per sample).
        """
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig) * eta
        # w_new += eta * neighborhood_function * (x-w)
        self.put('_weights', self.get('_weights') + jnp.einsum("ij, ijk->ijk", g, x - self.get('_weights')))

    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        self._check_input_len(data)
        winners_coords = jnp.argmin(self._distance_from_weights(data), axis=1)
        return self.get('_weights')[jnp.unravel_index(winners_coords, self.get('_weights').shape[:2])]

    def random_weights_init(self, data, seed=None):
        """Initializes the weights of the SOM
        picking random samples from data."""
        self._check_input_len(data)
        shape = self.get('_weights').shape
        weights = np.zeros(shape, dtype=np.float32)
        for i in range(shape[0]):
            for j in range(shape[1]):
                key, seed = random.split(seed)
                rand_i = random.randint(key, shape=(), minval=0, maxval=len(data) - 1)
                weights[i, j] = data[rand_i]
        self.put('_weights', jnp.asarray(weights))

    def pca_weights_init(self, data):
        """Initializes the weights to span the first two principal components.

        This initialization doesn't depend on random processes and
        makes the training process converge faster.

        It is strongly reccomended to normalize the data before initializing
        the weights and use the same normalization for the training data.
        """
        if self._input_len == 1:
            msg = "The data needs at least 2 features for pca initialization"
            raise ValueError(msg)
        self._check_input_len(data)
        if len(self.get('_neigx')) == 1 or len(self.get('_neigy')) == 1:
            msg = (
                "PCA initialization inappropriate:"
                + "One of the dimensions of the map is 1."
            )
            warn(msg)
        pc_length, eigvecs = jnp.linalg.eigh(jnp.cov(data))
        pc = jnp.matmul(eigvecs.T, data)
        pc_order = jnp.argsort(-pc_length)
        shape = self.get('_weights').shape
        weights = jnp.zeros(shape, dtype=jnp.float32)
        for i, c1 in enumerate(np.linspace(-1, 1, len(self.get('_neigx')))):
            for j, c2 in enumerate(np.linspace(-1, 1, len(self.get('_neigy')))):
                weights = weights.at[i, j].set(c1 * pc[pc_order[0]] + c2 * pc[pc_order[1]])
        self.put('_weights', weights)

    def train(self, data, num_iteration, random_key=None):
        """Trains the SOM.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_iteration : int
            If use_epochs is False, the weights will be
            updated num_iteration times. Otherwise they will be updated
            len(data)*num_iteration times.

        random_key : random seed
            If not None, samples are picked in random order.
            Otherwise the samples are picked sequentially.

        """
        # self._check_iteration_number(num_iteration)
        self._check_input_len(data)
        iterations = _build_iteration_indexes(
            len(data), len(num_iteration), self.verbose, random_key, self.use_epochs
        )
        if self.use_epochs:
            def get_decay_rate(iteration_index, data_len):
                return jnp.array(iteration_index / data_len, int)

        else:
            def get_decay_rate(iteration_index, data_len):
                return jnp.array(iteration_index, int)
        
        def update_step(t, iteration):
            decay_rate = get_decay_rate(t, len(data))
            self.update(
                data[iteration], self.winner(data[iteration]), decay_rate, len(num_iteration)
            )
            return t+1, None
        _, _ = nj.scan(update_step, jnp.array(0, int), iterations)
        
        metrics = {"quantization_error:": self.quantization_error(data),
                   "topographic_error_error": self.topographic_error(data)}
        return metrics

    def train_random(self, data, num_iteration, seed=None, verbose=False):
        """Trains the SOM picking samples at random from data.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_iteration : int
            Maximum number of iterations (one iteration per sample).

        verbose : bool (default=False)
            If True the status of the training
            will be printed at each time the weights are updated.
        """
        self.train(data, num_iteration, random_key=seed, verbose=verbose)

    def train_batch(self, data, num_iteration, verbose=False):
        """Trains the SOM using all the vectors in data sequentially.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        num_iteration : int
            Maximum number of iterations (one iteration per sample).

        verbose : bool (default=False)
            If True the status of the training
            will be printed at each time the weights are updated.
        """
        self.train(data, num_iteration, random_key=None, verbose=verbose)

    def distance_map(self, scaling="sum"):
        """Returns the distance map of the weights.
        If scaling is 'sum' (default), each cell is the normalised sum of
        the distances between a neuron and its neighbours. Note that this
        method uses the euclidean distance.

        Parameters
        ----------
        scaling : string (default='sum')
            If set to 'mean', each cell will be the normalized
            by the average of the distances of the neighbours.
            If set to 'sum', the normalization is done
            by the sum of the distances.
        """

        if scaling not in ["sum", "mean"]:
            raise ValueError(
                f'scaling should be either "sum" or "mean" (' f'"{scaling}" not valid)'
            )
        w_shape = self.get('_weights').shape
        um = np.nan * jnp.zeros((w_shape[0], w_shape[1], 8))  # 2 spots more for hexagonal topology

        ii = [[0, -1, -1, -1, 0, 1, 1, 1]] * 2
        jj = [[-1, -1, 0, 1, 1, 1, 0, -1]] * 2

        if self.topology == "hexagonal":
            ii = [[1, 1, 1, 0, -1, 0], [0, 1, 0, -1, -1, -1]]
            jj = [[1, 0, -1, -1, 0, 1], [1, 0, -1, -1, 0, 1]]

        for x in range(w_shape[0]):
            for y in range(w_shape[1]):
                w_2 = self.get('_weights')[x, y]
                e = y % 2 == 0  # only used on hexagonal topology
                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                    if (
                        x + i >= 0
                        and x + i < w_shape[0]
                        and y + j >= 0
                        and y + j < w_shape[1]
                    ):
                        w_1 = self.get('_weights')[x + i, y + j]
                        um = um.at[x, y, k].set(fast_norm(w_2 - w_1))

        if scaling == "mean":
            um = jnp.nanmean(um, axis=2)
        if scaling == "sum":
            um = jnp.nansum(um, axis=2)

        return um / um.max()

    def neighbor_dist(self, x, y):
        """Generate a distance vector that contains distances between
        the given neuron and all its neighbors (inf returned if the
        neighbor does not exist).

        Args:
            x (int): neuron index along first dimension
            y (int): neuron index along second dimension

        Returns:
            jnp.Array: distance array.
        """
        w_shape = self.get('_weights').shape
        if x<0 or x>=w_shape[0] or y<0 or y>=w_shape[1]:
            raise ValueError("Invalid neuron index")
            
        um = np.inf * jnp.ones((8,))

        ii = [[0, -1, -1, -1, 0, 1, 1, 1]] * 2
        jj = [[-1, -1, 0, 1, 1, 1, 0, -1]] * 2

        if self.topology == "hexagonal":
            ii = [[1, 1, 1, 0, -1, 0], [0, 1, 0, -1, -1, -1]]
            jj = [[1, 0, -1, -1, 0, 1], [1, 0, -1, -1, 0, 1]]

        w_2 = self.get('_weights')[x, y]
        e = y % 2 == 0  # only used on hexagonal topology
        neighbor_indices = []
        for k, (i, j) in enumerate(zip(ii[e], jj[e])):
            neighbor_indices.append((x + i, y + j))
            if (
                x + i >= 0
                and x + i < w_shape[0]
                and y + j >= 0
                and y + j < w_shape[1]
            ):
                w_1 = self.get('_weights')[x + i, y + j]
                um = um.at[k].set(fast_norm(w_2 - w_1))
        return um, neighbor_indices
    
    def get_neighbor_data(self, sample, data=None, min_samples=100):
        """Generate a tuple of data and distances from the given sample to
        the data from the winner neuron and its neighbors (if the number
        of samples in the winner neuron is less than the number of samples
        specified by min_samples).

        Args:
            sample (jnp.Array): 1D jax array that contains a data point
            data (jnp.Array, optional): entire dataset organized by the SOM. 
                Defaults to None.
            min_samples (int, optional): minimal number of samples to get from
                the SOM. Defaults to 1000.

        Returns:
            (candidate_data, dists): a list of data vectors and their distances
            from the given sample.
        """
        win = self.winner(sample)
        win = (win[0].tolist(), win[1].tolist())
        if data is None:
            assert len(self.winmap) > 0, "Must give data and run win_map before calling get_neighbor_data"
            candidate_data = self.winmap[win][:]
        else:
            self.winmap = self.win_map(data, return_indices=False)
            candidate_data = self.winmap[win][:]
        if len(candidate_data) < min_samples:
            neighbor_dists, neighbor_indices = self.neighbor_dist(win[0], win[1])
            mask = jnp.where(neighbor_dists != jnp.inf)[0]
            neighbor_tuples = list(zip(neighbor_dists, neighbor_indices))
            neighbor_tuples = [neighbor_tuples[i] for i in mask.tolist()]
            neighbor_tuples.sort(key=lambda neighbor: neighbor[0])
            i = 0
            while len(candidate_data) < min_samples and i < len(neighbor_tuples):
                neighbor_id = neighbor_tuples[i][1]
                candidate_data.extend(self.winmap[neighbor_id])
                i += 1
        if len(candidate_data) < min_samples:
            raise ValueError(
                f"Insufficient number of samples in winner neuron {win} and neighbors to get {min_samples} samples"
            )
        dists = jnp.linalg.norm(jnp.asarray(candidate_data) - sample, axis=-1)
        return candidate_data, dists
    
    def get_neighbor_indices(self, sample, data, indices, winmap_idx=None, winmap_data=None, min_samples=100):
        """Generate a tuple of indices and distances from the given sample to
        the data from the winner neuron and its neighbors (if the number
        of samples in the winner neuron is less than the number of samples
        specified by min_samples).

        Args:
            sample (jnp.Array): 1D jax array that contains a data point
            data (jnp.Array): 2D entire dataset as an array.
            indices (jnp.Array): array of int64 indices that used to index
                the entire dataset. 1D array, which aligns with the data
            min_samples (int, optional): minimal number of samples to get using
                the SOM. Defaults to 1000.

        Returns:
            candidate_idx: (list): 1D shape: List[data_id]
            dists: (jax.Array): 1d shape: jax.Array: (N_data_id,) List[distance]
            from the given sample.
            winmap_idx: (dict): {(i, j): [data_id]}
            winmap_data: (dict): {(i, j): [embedding]}
        """
        win = tuple(np.asarray(self.winner(sample)))
        if winmap_idx is None or winmap_data is None:
            winmap_idx, winmap_data = self.win_map_index_n_data(indices, data)
        candidate_data = winmap_data[win][:]
        candidate_idx = winmap_idx[win][:]
        if len(candidate_data) < min_samples:
            neighbor_dists, neighbor_indices = self.neighbor_dist(win[0], win[1])
            mask = jnp.where(neighbor_dists != jnp.inf)[0]
            neighbor_tuples = list(zip(neighbor_dists, neighbor_indices))
            neighbor_tuples = [neighbor_tuples[i] for i in mask.tolist()]
            neighbor_tuples.sort(key=lambda neighbor: neighbor[0])
            i = 0
            while len(candidate_data) < min_samples and i < len(neighbor_tuples):
                neighbor_id = neighbor_tuples[i][1]
                candidate_data.extend(winmap_data[neighbor_id])
                candidate_idx.extend(winmap_idx[neighbor_id])
                i += 1
        if len(candidate_data) < min_samples:
            raise ValueError(
                f"Insufficient number of samples in winner neuron {win} and neighbors to get {min_samples} samples"
            )
        dists = jnp.linalg.norm(jnp.asarray(candidate_data) - sample, axis=-1)
        return candidate_idx, dists, winmap_idx, winmap_data

    def activation_response(self, data):
        """
        Returns a matrix where the element i,j is the number of times
        that the neuron i,j have been winner.
        """
        self._check_input_len(data)
        w_shape = self.get('_weights').shape
        a = jnp.zeros((w_shape[0], w_shape[1]))
        for x in data:
            a = a.at[self.winner(x)].add(1)
        return a

    def _distance_from_weights(self, data):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        input_data = jnp.asarray(data)
        weights_flat = self.get('_weights').reshape(-1, self.get('_weights').shape[2])
        input_data_sq = jnp.power(input_data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = jnp.power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = jnp.dot(input_data, weights_flat.T)
        return jnp.sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        self._check_input_len(data)
        return jnp.linalg.norm(data - self.quantization(data), axis=1).mean()

    def topographic_error(self, data):
        """Returns the topographic error computed by finding
        the best-matching and second-best-matching neuron in the map
        for each input and then evaluating the positions.

        A sample for which these two nodes are not adjacent counts as
        an error. The topographic error is given by the
        the total number of errors divided by the total of samples.

        If the topographic error is 0, no error occurred.
        If 1, the topology was not preserved for any of the samples."""
        self._check_input_len(data)
        total_neurons = np.prod(self.get('_activation_map').shape)
        if total_neurons == 1:
            warn("The topographic error is not defined for a 1-by-1 map.")
            return np.nan
        if self.topology == "hexagonal":
            return self._topographic_error_hexagonal(data)
        else:
            return self._topographic_error_rectangular(data)

    def _topographic_error_hexagonal(self, data):
        """Return the topographic error for hexagonal grid"""
        b2mu_inds = jnp.argsort(self._distance_from_weights(data), axis=1)[:, :2]
        b2mu_coords = [
            [
                self._get_euclidean_coordinates_from_index(bmu[0]),
                self._get_euclidean_coordinates_from_index(bmu[1]),
            ]
            for bmu in b2mu_inds
        ]
        b2mu_coords = jnp.asarray(b2mu_coords)
        b2mu_neighbors = [
            jnp.isclose(1, jnp.linalg.norm(bmu1 - bmu2)) for bmu1, bmu2 in b2mu_coords
        ]
        te = 1 - jnp.mean(jnp.array(b2mu_neighbors))
        return te

    def _topographic_error_rectangular(self, data):
        """Return the topographic error for rectangular grid"""
        t = 1.42
        # b2mu: best 2 matching units
        b2mu_inds = jnp.argsort(self._distance_from_weights(data), axis=1)[:, :2]
        b2my_xy = jnp.unravel_index(b2mu_inds, self.get('_weights').shape[:2])
        b2mu_x, b2mu_y = b2my_xy[0], b2my_xy[1]
        dxdy = jnp.hstack([jnp.diff(b2mu_x), jnp.diff(b2mu_y)])
        distance = jnp.linalg.norm(dxdy, axis=1)
        return (distance > t).mean()

    def _get_euclidean_coordinates_from_index(self, index):
        """Returns the Euclidean coordinated of a neuron using its
        index as the input"""
        if index < 0:
            return (-1, -1)
        y = self.get('_weights').shape[1]
        coords = self.convert_map_to_euclidean((int(index / y), index % y))
        return coords

    def win_map(self, data, return_indices=False):
        """Returns a dictionary wm where wm[(i,j)] is a list with:
        - all the patterns that have been mapped to the position (i,j),
          if return_indices=False (default)
        - all indices of the elements that have been mapped to the
          position (i,j) if return_indices=True"""
        self._check_input_len(data)
        winmap = defaultdict(list)
        for i, x in enumerate(data):
            winmap[tuple(np.asarray(self.winner(x)))].append(i if return_indices else x)
        return winmap
    
    def win_map_index_n_data(self, indices, data):
        """Returns a dictionary wm where wm[(i,j)] is a list of indices.

        Args:
            indices (jnp.Array): (N,) shaped 1D array of int64 typed 
                index for data
            data (jnp.Array): (N, input_len) shaped data array 

        Returns:
            dict: winmap_idx {node_index : list of int64 index}
            dict: winmap_data  {node_index : list of data points}
        """
        self._check_input_len(data)
        winmap_data = defaultdict(list)
        winmap_idx = defaultdict(list)
        # for x, i in zip(data, indices):
        #     winmap[self.winner(x)].append(i)
        v_winner = jax.vmap(self.winner)
        winners = v_winner(data)  # tuple of (2,), each element has (N,) vector 
        winners = jnp.vstack(winners).swapaxes(0, 1) # (N, 2)
        w_shape = self.get('_weights').shape
        ids = jnp.empty((w_shape[0], w_shape[1], 2), dtype=jnp.int32)  # (x, y, 2)
        ids = ids.at[:,:,0].set(jnp.arange(w_shape[0])[:, None])  # Row indices
        ids = ids.at[:,:,1].set(jnp.arange(w_shape[1]))
        ids = ids.reshape((-1, 2))   # (x*y, 2)
        def winner_mask(winner, index):
            return jnp.all(winner == index)
        v_winner_mask = jax.vmap(winner_mask, in_axes=(None, 0))  # 1 winner on all indices
        vv_winner_mask = jax.vmap(v_winner_mask, in_axes=(0, None))  # all winners on all indices
        mask_tensor = vv_winner_mask(winners, ids)  # (N, x*y)
        ids = ids.tolist()
        for i in range(len(ids)):
            winmap_idx[tuple(ids[i])] = list(indices[mask_tensor[:, i]])
        for i in range(len(ids)):
            winmap_data[tuple(ids[i])] = list(data[mask_tensor[:, i]])

        return winmap_idx, winmap_data

    def labels_map(self, data, labels):
        """Returns a dictionary wm where wm[(i,j)] is a dictionary
        that contains the number of samples from a given label
        that have been mapped in position i,j.

        Parameters
        ----------
        data : np.array or list
            Data matrix.

        label : np.array or list
            Labels for each sample in data.
        """
        self._check_input_len(data)
        if not len(data) == len(labels):
            raise ValueError("data and labels must have the same length.")
        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[tuple(np.asarray(self.winner(x)))].append(l)
        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap

if __name__ == "__main__":
    import pprint
    seed = 42
    x, y, length = 9, 9, 7
    n_samples = 210
    som = MiniSom(x, y, length, name='testsom', verbose=False, use_epochs=True)
    data = random.uniform(random.key(seed), (n_samples, length))
    iterations = jnp.arange(1000)
    # p_init_weight = jax.jit(nj.pure(som.init_weights))
    params = nj.init(som.init_weights)({}, random.key(1), seed=0)
    pprint.pprint(f"global params after init_weights: {params.keys()}")
    params = nj.init(som.pca_weights_init)(params, data, seed=0)
    pprint.pprint(f"global params after pca_weights_init: {params.keys()}")
    params = nj.init(som.train)(params, data, iterations, random.key(seed), seed=0)
    pprint.pprint(f"global params after init train: {params.keys()}")
    p_train = jax.jit(nj.pure(som.train))
    print("======= After purify the train func. ========")
    params, outputs = p_train(params, data, iterations, random.key(seed), seed=0)
    pprint.pprint(f"params shape: {jax.tree_map(lambda x: x.shape, params)}")
    pprint.pprint(outputs)
    p_winner = jax.jit(nj.pure(som.winner))
    params, win = p_winner(params, data[2], seed=0)
    print(f"the winner is {win}")
    # we cannot jit the get_neighbor_data() function since it turns jax array back
    # to tuple of ints in order to be used as keys in dict.
    p_get_neighbor_data = nj.pure(som.get_neighbor_data)
    params, (candidate_data, dists) = p_get_neighbor_data(params, data[10], data, min_samples=10)
    pprint.pprint(f"distances {dists}")
    p_win_map_index_n_data = nj.pure(som.win_map_index_n_data)
    params, (winmap_idx, winmap_data) = p_win_map_index_n_data(params, jnp.arange(len(data)), data)
    pprint.pprint(f"winmap_idx {winmap_idx}")   #\nwinmap_data {winmap_data}")
    p_get_neighbor_indices = nj.pure(som.get_neighbor_indices)
    params, (candidate_idx, dists2, _, _) = p_get_neighbor_indices(params, data[10], data, jnp.arange(len(data)), min_samples=10)
    pprint.pprint(f"sampled indices {candidate_idx} \ndistances {dists2}")


