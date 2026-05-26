"""
Generative Quantum Eigensolver (GQE) Main Implementation

Orchestrates the hybrid classical-quantum optimization loop for ground state
energy estimation using a generative model to design quantum circuits.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

from quantum_simulator import QuantumSimulator, exact_ground_state_energy
from generative_model import (QuantumCircuitGenerativeModel,
                             ParameterizedCircuitModel,
                             AdaptiveGenerativeModel,
                             create_default_system_properties)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerativeQuantumEigensolver:
    """
    Generative Quantum Eigensolver (GQE)
    
    Combines a classical generative neural network with quantum simulation
    to estimate ground state energies of quantum systems.
    
    Algorithm:
    1. Generative model (NN) takes system properties
    2. Model outputs quantum circuit parameters
    3. Quantum simulator evaluates circuit energy
    4. Loss computed as difference from target
    5. NN updated via backpropagation
    6. Repeat until convergence
    """
    
    def __init__(self, num_qubits: int, num_layers: int = 2,
                 hidden_dim: int = 64, learning_rate: float = 0.01,
                 entanglement: str = 'linear', adaptive: bool = False,
                 verbose: bool = True):
        """
        Initialize GQE solver.
        
        Args:
            num_qubits: Number of qubits
            num_layers: Number of circuit layers
            hidden_dim: Hidden dimension for neural network
            learning_rate: Optimizer learning rate
            entanglement: Type of entanglement ('none', 'linear', 'full')
            adaptive: Use adaptive model that scales with system size
            verbose: Print detailed logging
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.entanglement = entanglement
        self.verbose = verbose
        
        # Initialize quantum simulator
        self.simulator = QuantumSimulator(num_qubits, backend='statevector')
        
        # Initialize generative model
        if adaptive:
            self.model = AdaptiveGenerativeModel(
                num_qubits=num_qubits,
                num_layers=num_layers,
                learning_rate=learning_rate
            )
        else:
            self.model = QuantumCircuitGenerativeModel(
                num_qubits=num_qubits,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate
            )
        
        # Initialize circuit builder
        self.circuit_model = ParameterizedCircuitModel(
            num_qubits=num_qubits,
            num_layers=num_layers,
            entanglement=entanglement
        )
        
        # Tracking
        self.history = defaultdict(list)
        self.best_energy = float('inf')
        self.best_parameters = None
        
        if self.verbose:
            logger.info(f"Initialized GQE: qubits={num_qubits}, "
                       f"layers={num_layers}, entanglement={entanglement}")
    
    def optimize(self, hamiltonian: np.ndarray, max_iterations: int = 100,
                 target_accuracy: float = 1.6e-3, batch_size: int = 1,
                 adaptive_lr: bool = True) -> Dict:
        """
        Run GQE optimization.
        
        Args:
            hamiltonian: Hamiltonian matrix (2^n x 2^n)
            max_iterations: Maximum optimization iterations
            target_accuracy: Target accuracy in mHa (1.6 mHa ≈ chemical accuracy)
            batch_size: Batch size for training
            adaptive_lr: Reduce learning rate on plateau
            
        Returns:
            Dictionary with results
        """
        # Get exact ground state for reference
        exact_energy, exact_state = exact_ground_state_energy(hamiltonian)
        
        if self.verbose:
            logger.info(f"Exact ground state energy: {exact_energy:.6f} Ha")
        
        # System properties (simplified)
        system_props = create_default_system_properties(self.num_qubits)
        
        # Training loop
        patience = 0
        max_patience = 10
        
        for iteration in range(max_iterations):
            # Generate circuit parameters
            circuit_params = self.model.generate_circuit_parameters(system_props)
            
            # Build and evaluate circuit
            circuit = self.circuit_model.build_circuit(circuit_params, self.simulator)
            current_energy = self.simulator.compute_expectation_value(
                circuit, hamiltonian
            )
            
            # Compute loss
            energy_error = current_energy - exact_energy
            loss = torch.tensor(energy_error ** 2, requires_grad=True)
            
            # Update model
            self.model.update(loss)
            
            # Track history
            abs_error_mha = abs(energy_error) * 1000  # Convert to mHa
            self.history['energy'].append(current_energy)
            self.history['error'].append(abs_error_mha)
            self.history['loss'].append(float(loss.detach()))
            
            # Update best
            if current_energy < self.best_energy:
                self.best_energy = current_energy
                self.best_parameters = circuit_params.copy()
                patience = 0
            else:
                patience += 1
            
            # Logging
            if self.verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
                circuit_depth = self.simulator.get_circuit_depth(circuit)
                status = "✓" if abs_error_mha <= target_accuracy else " "
                logger.info(
                    f"Iter {iteration+1:3d}: E={current_energy:10.6f} Ha, "
                    f"Error={abs_error_mha:8.4f} mHa, "
                    f"Depth={circuit_depth:2d} {status}"
                )
            
            # Check convergence
            if abs_error_mha <= target_accuracy:
                if self.verbose:
                    logger.info(f"✓ Target accuracy achieved!")
                break
            
            # Adaptive learning rate
            if adaptive_lr and patience > max_patience:
                new_lr = self.learning_rate * 0.5
                self.model.set_learning_rate(new_lr)
                patience = 0
                if self.verbose:
                    logger.info(f"Learning rate reduced to {new_lr}")
        
        # Build best circuit
        best_circuit = self.circuit_model.build_circuit(
            self.best_parameters, self.simulator
        )
        
        # Compile results
        results = {
            'energy': self.best_energy,
            'exact_energy': exact_energy,
            'error': abs(self.best_energy - exact_energy),
            'error_mha': abs(self.best_energy - exact_energy) * 1000,
            'chemical_accuracy': abs(self.best_energy - exact_energy) * 1000 < target_accuracy,
            'circuit_depth': self.simulator.get_circuit_depth(best_circuit),
            'circuit_gates': self.simulator.get_gate_count(best_circuit),
            'iterations': len(self.history['energy']),
            'history': dict(self.history),
            'parameters': self.best_parameters
        }
        
        return results
    
    def get_history(self) -> Dict:
        """Get optimization history."""
        return dict(self.history)
    
    def get_best_circuit(self):
        """Get the best circuit found."""
        if self.best_parameters is None:
            raise ValueError("No optimization has been run yet")
        
        return self.circuit_model.build_circuit(
            self.best_parameters, self.simulator
        )
    
    def evaluate_circuit(self, parameters: np.ndarray,
                        hamiltonian: np.ndarray) -> float:
        """
        Evaluate energy for given circuit parameters.
        
        Args:
            parameters: Circuit parameters
            hamiltonian: Hamiltonian matrix
            
        Returns:
            Energy expectation value
        """
        circuit = self.circuit_model.build_circuit(parameters, self.simulator)
        return self.simulator.compute_expectation_value(circuit, hamiltonian)
    
    def scan_parameters(self, hamiltonian: np.ndarray,
                       param_range: Tuple[float, float],
                       param_index: Tuple[int, int, int],
                       num_points: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scan a single parameter to visualize the energy landscape.
        
        Args:
            hamiltonian: Hamiltonian matrix
            param_range: (min, max) values for parameter
            param_index: (layer, qubit, component) indices
            num_points: Number of points to scan
            
        Returns:
            (parameter_values, energies) arrays
        """
        param_values = np.linspace(param_range[0], param_range[1], num_points)
        energies = []
        
        for param_val in param_values:
            params = self.best_parameters.copy()
            layer, qubit, comp = param_index
            params[layer, qubit, comp] = param_val
            
            energy = self.evaluate_circuit(params, hamiltonian)
            energies.append(energy)
        
        return param_values, np.array(energies)


class VQEBaseline:
    """
    Variational Quantum Eigensolver (VQE) baseline for comparison.
    
    Standard VQE with fixed ansatz for benchmarking against GQE.
    """
    
    def __init__(self, num_qubits: int, num_layers: int = 2):
        """Initialize VQE baseline."""
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.simulator = QuantumSimulator(num_qubits)
        self.circuit_model = ParameterizedCircuitModel(
            num_qubits, num_layers, entanglement='linear'
        )
    
    def optimize(self, hamiltonian: np.ndarray,
                 max_iterations: int = 100) -> Dict:
        """
        Run VQE optimization with fixed ansatz.
        
        Args:
            hamiltonian: Hamiltonian matrix
            max_iterations: Maximum iterations
            
        Returns:
            Results dictionary
        """
        exact_energy, _ = exact_ground_state_energy(hamiltonian)
        
        # Initialize random parameters
        parameters = np.random.uniform(0, 2*np.pi, 
                                      (self.num_layers, self.num_qubits, 2))
        
        learning_rate = 0.01
        best_energy = float('inf')
        best_params = parameters.copy()
        
        for iteration in range(max_iterations):
            # Evaluate current parameters
            circuit = self.circuit_model.build_circuit(parameters, self.simulator)
            energy = self.simulator.compute_expectation_value(circuit, hamiltonian)
            
            if energy < best_energy:
                best_energy = energy
                best_params = parameters.copy()
            
            # Numerical gradient (finite differences)
            gradient = np.zeros_like(parameters)
            epsilon = 1e-4
            
            for i in range(parameters.shape[0]):
                for j in range(parameters.shape[1]):
                    for k in range(parameters.shape[2]):
                        # Forward difference
                        parameters[i, j, k] += epsilon
                        circuit_plus = self.circuit_model.build_circuit(
                            parameters, self.simulator
                        )
                        energy_plus = self.simulator.compute_expectation_value(
                            circuit_plus, hamiltonian
                        )
                        
                        # Backward difference
                        parameters[i, j, k] -= 2 * epsilon
                        circuit_minus = self.circuit_model.build_circuit(
                            parameters, self.simulator
                        )
                        energy_minus = self.simulator.compute_expectation_value(
                            circuit_minus, hamiltonian
                        )
                        
                        # Gradient
                        gradient[i, j, k] = (energy_plus - energy_minus) / (2 * epsilon)
                        
                        # Reset
                        parameters[i, j, k] += epsilon
            
            # Update parameters
            parameters -= learning_rate * gradient
            
            # Adaptive learning rate
            if iteration % 20 == 0:
                learning_rate *= 0.95
        
        return {
            'energy': best_energy,
            'exact_energy': exact_energy,
            'error': abs(best_energy - exact_energy),
            'error_mha': abs(best_energy - exact_energy) * 1000,
            'iterations': max_iterations
        }
