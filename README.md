# TORCWA

***TORCHWA*** (**torc**h + **rcwa**) is a PyTorch implementation of rigorous coupled-wave analysis (RCWA)

## Table of Contents

- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Usage](#usage)

## Key Features

- **GPU-accelerated** simulation
- Supporting **automatic differentiation** for optimization
- Units: Lorentz-Heaviside units
	* Speed of light: 1
	* Permittivity and permeability of vacuum: both 1
- Notation: exp(-*jωt*)

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- Required libraries (listed in `requirements.txt`)

### Installation

Clone the repository:

```bash
git clone https://github.com/Alexin-CH/TORCWA.git
cd TORCWA
```

Install the required dependencies:

``` bash
make
```

## Usage

Main script is located in the `src` directory.
