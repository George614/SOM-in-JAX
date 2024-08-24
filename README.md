# SOM-in-JAX
Self-Organizing Maps implemented in JAX

## Overview

This repository contains an implementation of Self-Organizing Maps (SOM) using JAX, a high-performance numerical computing library. The SOM algorithm is an unsupervised learning technique used for dimensionality reduction and data visualization.


## Features

- Efficient implementation using JAX for GPU acceleration
- Support for both rectangular and hexagonal topologies
- Various neighborhood functions: Gaussian, Mexican hat, bubble, and triangle
- Multiple distance metrics: Euclidean, cosine, Manhattan, and Chebyshev
- Customizable learning rate and decay functions
- Quantization error and topographic error calculations
- PCA-based weight initialization
- Batch and random training modes


## Installation

To use this project, you'll need to have JAX and its dependencies installed. You can install them using pip:
`pip install jax jaxlib`

Clone this repository:
```
git clone https://github.com/yourusername/SOM-in-JAX.git
cd SOM-in-JAX
```

## Usage

Here's a basic example of how to use the MiniSom class:
```
import jax.numpy as jnp
from jax import random
from node import MiniSom

# Initialize SOM
som = MiniSom(10, 10, 3, sigma=1.0, learning_rate=0.5, random_seed=42)

# Generate random data
key = random.PRNGKey(42)
data = random.normal(key, (1000, 3))

# Train the SOM
som.train(data, 100)

# Get the winner neuron for a sample
sample = jnp.array([1.0, 0.5, -0.5])
winner = som.winner(sample)
print(f"Winner neuron coordinates: {winner}")
```

For more advanced usage and examples, please refer to the code comments and docstrings in the `node.py` file. The main implementation of the SOM algorithm can be found in the `MiniSom` class.  This class provides methods for initializing the SOM, training it on data, and querying the trained map.


## NinjaX Integration

The repository also includes an implementation using NinjaX, a library for managing state in JAX. The `MiniSom` class using NinjaX can be found in `node_nj.py`. 
This implementation provides better state management and allows for easier integration with other JAX-based libraries.


## Testing

The repository includes a comprehensive test suite to ensure the correctness of the SOM implementation. You can find the tests in the `TestMinisom` class.

To run the tests, you can use a Python test runner like pytest.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements
This implementation is inspired by the original MiniSom project and adapted for use with JAX.

## TODO
- Add more examples and use cases
- Implement visualization tools for the trained SOM
- Optimize performance for large-scale datasets
- Add support for custom distance functions

For any questions or issues, please open an GitHub issue in this repository.