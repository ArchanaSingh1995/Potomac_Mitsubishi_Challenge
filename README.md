# GQE Prototype Implementation

This folder contains a reference implementation of the **Generative Quantum Eigensolver (GQE)** for the Mitsubishi Chemical Group / AIST quantum materials discovery challenge.

## 📁 Folder Structure

```
gqe_prototype/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── gqe_prototype.py            # Main GQE implementation
├── generative_model.py         # Generative model (neural network)
├── quantum_simulator.py        # Quantum circuit simulation
├── benchmark.py                # Benchmarking utilities
├── examples/
│   ├── h2_ground_state.py      # Example: H2 molecule
│   ├── lih_ground_state.py     # Example: LiH molecule
│   └── h2o_ground_state.py     # Example: H2O molecule
└── results/                    # Output directory for results
```

## 🚀 Quick Start

### Installation

```bash
cd gqe_prototype
pip install -r requirements.txt
```

### Running a Simple Example

```bash
python examples/h2_ground_state.py
```

This will:
1. Set up H₂ molecule quantum system
2. Train a generative model to create quantum circuits
3. Evaluate ground state energy using GQE
4. Compare with exact solution
5. Display accuracy metrics

## 📚 Key Components

### 1. **Generative Model** (`generative_model.py`)
- Neural network that learns to generate parameterized quantum circuits
- Input: Molecular/system properties
- Output: Circuit gate parameters and structure
- Training: Iteratively improved based on energy evaluation

**Key Features:**
- Flexible circuit architecture
- Trainable gate parameters
- Support for multi-layer circuits

### 2. **Quantum Simulator** (`quantum_simulator.py`)
- Handles quantum circuit simulation
- Uses state vector simulation (Qiskit backend)
- Computes ground state energy
- Supports multiple simulation methods

**Key Features:**
- Circuit construction and execution
- Energy measurement via Hamiltonian expectation values
- Noise-aware simulation (optional)

### 3. **GQE Main Loop** (`gqe_prototype.py`)
- Orchestrates the hybrid classical-quantum workflow
- Implements training loop
- Manages convergence and optimization
- Provides detailed logging

**Key Features:**
- Hybrid optimization (classical + quantum)
- Convergence tracking
- Result logging and visualization

### 4. **Benchmarking** (`benchmark.py`)
- Compare GQE against:
  - Exact diagonalization (ground truth)
  - Variational Quantum Eigensolver (VQE)
  - Classical methods
- Calculate metrics:
  - Absolute energy error
  - Chemical accuracy (mHa)
  - Circuit depth
  - Training time

## 🔧 How to Extend This Code

### Adding a New Target Molecule

1. Create a new file in `examples/`
2. Define the molecular system:

```python
from gqe_prototype import GQE
from quantum_simulator import QuantumSimulator
import numpy as np

# Define your molecule
molecule_name = "Your_Molecule"
num_qubits = 4  # Depends on your system
hamiltonian = get_hamiltonian_for_molecule()

# Create GQE solver
gqe = GQE(
    num_qubits=num_qubits,
    num_layers=2,
    learning_rate=0.01
)

# Run optimization
result = gqe.optimize(
    hamiltonian=hamiltonian,
    max_iterations=100,
    target_accuracy=1.6e-3  # mHa
)

print(f"Ground state energy: {result['energy']}")
print(f"Accuracy: {result['error']} mHa")
```

### Customizing the Generative Model

Edit `generative_model.py`:

```python
class CustomGenerativeModel(GenerativeModel):
    def __init__(self, num_qubits, model_depth):
        super().__init__(num_qubits)
        # Your custom architecture here
        self.custom_layers = ...
    
    def generate_circuit(self, input_features):
        # Your circuit generation logic
        return circuit_parameters
```

### Adding Custom Hamiltonian

```python
def get_custom_hamiltonian(num_qubits):
    """Define your custom Hamiltonian"""
    H = np.zeros((2**num_qubits, 2**num_qubits))
    # Build Hamiltonian matrix
    return H
```

## 📊 Understanding the Output

When you run an example, you'll see:

```
GQE Optimization Progress
========================
Iteration 1: Energy = -1.042 Ha, Accuracy = 0.156 mHa
Iteration 2: Energy = -1.085 Ha, Accuracy = 0.113 mHa
Iteration 3: Energy = -1.101 Ha, Accuracy = 0.025 mHa
...
Iteration 50: Energy = -1.1026 Ha, Accuracy = 0.0018 mHa ✓

Final Results:
- Ground state energy: -1.1026 Ha
- Exact energy: -1.1026 Ha
- Absolute error: 0.0018 mHa
- Chemical accuracy achieved: YES ✓
- Circuit depth: 12
- Training time: 2.34 seconds
```

### Key Metrics Explained

- **Energy (Ha)**: Hartree units - the main optimization target
- **Accuracy (mHa)**: Millihartree - how close to exact solution
- **Chemical Accuracy**: < 1.6 mHa is considered acceptable for chemistry
- **Circuit Depth**: Number of gate layers (important for scalability)
- **Training Time**: Total optimization time

## 🧪 Experimental Workflow

### Phase 1: Validate with Small Systems (8-12 qubits)

```bash
python examples/h2_ground_state.py      # ~4 qubits
python examples/lih_ground_state.py     # ~8-10 qubits
```

### Phase 2: Test Circuit Scaling

Modify examples to increase `num_qubits` and `num_layers`:

```python
# In your example file
gqe = GQE(
    num_qubits=20,        # Increase from 4 to 20
    num_layers=4,         # Increase circuit depth
    learning_rate=0.01
)
```

Monitor:
- Training time
- Memory usage
- Convergence behavior
- Final accuracy

### Phase 3: Optimization & Benchmarking

```python
from benchmark import benchmark_gqe, compare_with_vqe

# Run comprehensive benchmark
results = benchmark_gqe(
    num_qubits_range=[4, 8, 12, 16, 20],
    num_trials=5,
    save_results=True
)

# Compare with VQE
comparison = compare_with_vqe(
    molecule="H2",
    gqe_config=gqe_config,
    vqe_config=vqe_config
)
```

## 🎯 Implementation Roadmap

### Current State (MVP)
- [x] Basic GQE for 4 qubits (H₂)
- [x] Simple neural network generative model
- [x] Energy evaluation using state vector simulation
- [x] Convergence tracking
- [x] Benchmarking against exact solution

### Phase 1: Extend to Medium Systems
- [ ] Support for 12-16 qubits
- [ ] Optimized circuit structures
- [ ] Advanced optimization algorithms (Adam, RMSprop)
- [ ] Noise simulation (depolarizing, amplitude damping)

### Phase 2: Scale to Large Systems
- [ ] 20-30 qubit systems
- [ ] GPU-accelerated simulation (CUDA-Q integration)
- [ ] Advanced circuit ansätze
- [ ] Tensor network methods

### Phase 3: Production & Benchmarking
- [ ] Full VQE comparison
- [ ] Multiple molecular systems
- [ ] Comprehensive benchmarking suite
- [ ] Publication-ready results

## 📖 Usage Examples

### Example 1: Basic Ground State Energy Calculation

```python
from gqe_prototype import GQE
from quantum_simulator import QuantumSimulator

# Setup
gqe = GQE(num_qubits=4, num_layers=2)

# Run optimization
result = gqe.optimize(
    hamiltonian=H2_hamiltonian,
    max_iterations=100
)

# Print results
print(f"Energy: {result['energy']:.6f} Ha")
print(f"Error: {result['error']:.6f} mHa")
```

### Example 2: Scaling Study

```python
from benchmark import scaling_study

# Test scalability from 4 to 20 qubits
results = scaling_study(
    qubit_range=[4, 8, 12, 16, 20],
    max_iterations=100
)

# Analyze results
for qubits, data in results.items():
    print(f"{qubits} qubits: Energy={data['energy']:.6f}, Time={data['time']:.2f}s")
```

### Example 3: Compare with VQE

```python
from benchmark import compare_algorithms

comparison = compare_algorithms(
    algorithm1=('GQE', gqe_instance),
    algorithm2=('VQE', vqe_instance),
    hamiltonian=H2_hamiltonian
)

print(f"GQE Error: {comparison['gqe_error']:.6f} mHa")
print(f"VQE Error: {comparison['vqe_error']:.6f} mHa")
print(f"GQE is {comparison['speedup']:.2f}x faster")
```

## 🔬 Scientific Concepts

### Generative Quantum Eigensolver

The GQE approach differs from standard VQE:

**VQE (Variational Quantum Eigensolver):**
```
Circuit Architecture → Fixed
Parameters → Optimized
Challenge: Finding good ansatz is hard
```

**GQE (Generative Quantum Eigensolver):**
```
Generative Model (NN) → Learns circuit structure
Parameters → Generated by model
Advantage: More flexible, learns good structures
```

### Hybrid Workflow

1. **Classical Phase:**
   - Generative model takes system properties
   - Outputs circuit parameters
   
2. **Quantum Phase:**
   - Simulate circuit on quantum system
   - Measure expectation value of Hamiltonian
   
3. **Feedback:**
   - Compute loss (difference from target)
   - Update generative model using gradient descent
   
4. **Repeat** until convergence

## 📝 Contributing & Modifications

When extending this code:

1. **Add docstrings** to all new functions
2. **Include type hints** for better code clarity
3. **Write tests** for new components
4. **Document** any new parameters
5. **Update this README** with new features

## 🐛 Common Issues & Solutions

### Issue: Training converges slowly

**Solution:** Increase learning rate, try different optimizers

```python
gqe = GQE(
    num_qubits=4,
    learning_rate=0.1,  # Increase from 0.01
    optimizer='adam'     # Try different optimizer
)
```

### Issue: Memory usage grows with qubits

**Solution:** Use a different simulator backend or batch evaluation

```python
simulator = QuantumSimulator(
    backend='qiskit_aer',  # Lower memory overhead
    max_memory_mb=2048
)
```

### Issue: Results don't match expectations

**Solution:** Validate Hamiltonian, check circuit construction

```python
# Verify Hamiltonian
H_eigenvalues = np.linalg.eigvalsh(hamiltonian)
print(f"Ground state energy (exact): {H_eigenvalues[0]}")
```

## 📚 References & Further Reading

- **GQE Papers:** Search for "Generative Quantum Eigensolver"
- **VQE Review:** "The Variational Quantum Eigensolver" (Cao et al.)
- **Quantum Circuits:** "An Introduction to Quantum Machine Learning" (Schuld & Petruccione)
- **NVIDIA CUDA-Q:** https://developer.nvidia.com/cuda-q

## 📞 Support

For questions about:
- **Quantum algorithms:** Review quantum computing literature
- **Implementation details:** Check inline code comments
- **Challenge requirements:** Refer to main challenge PDF

---

**Next Steps:**
1. Install dependencies: `pip install -r requirements.txt`
2. Run first example: `python examples/h2_ground_state.py`
3. Modify and extend for your target molecule
4. Benchmark and optimize for scalability
