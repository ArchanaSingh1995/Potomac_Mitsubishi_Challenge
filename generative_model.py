"""
Generative Model Module

Neural network-based generative model for learning quantum circuit structures
and parameters. The model takes system properties and generates optimized
circuit parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class QuantumCircuitGenerativeModel(nn.Module):
    """
    Neural network that generates quantum circuit parameters.
    
    Architecture:
    - Input layer: System/molecular properties
    - Hidden layers: Learn latent representations
    - Output layer: Circuit parameters for each layer
    """
    
    def __init__(self, num_qubits: int, num_layers: int = 2,
                 hidden_dim: int = 64, learning_rate: float = 0.01):
        """
        Initialize the generative model.
        
        Args:
            num_qubits: Number of qubits in target circuit
            num_layers: Number of circuit layers to generate
            hidden_dim: Hidden layer dimension
            learning_rate: Optimizer learning rate
        """
        super().__init__()
        
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Input features: molecular properties (simplified)
        # E.g., bond length, atomic charges, etc.
        input_dim = 4  # Can be extended for more properties
        
        # Total output: parameters for RX and RY layers
        # Each layer has num_qubits rotations * 2 (RX and RY)
        output_dim = num_layers * num_qubits * 2
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Output activation to constrain parameters to [0, 2π]
        self.output_scaling = 2 * np.pi
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        logger.info(f"Initialized GenerativeModel: {num_qubits} qubits, "
                   f"{num_layers} layers, hidden_dim={hidden_dim}")
    
    def forward(self, system_properties: torch.Tensor) -> torch.Tensor:
        """
        Generate circuit parameters from system properties.
        
        Args:
            system_properties: Tensor of shape (batch_size, input_features)
            
        Returns:
            Circuit parameters tensor of shape (batch_size, num_layers, num_qubits, 2)
        """
        # Pass through network
        batch_size = system_properties.shape[0]
        output = self.network(system_properties)
        
        # Scale to [0, 2π]
        parameters = torch.sigmoid(output) * self.output_scaling
        
        # Reshape to (batch_size, num_layers, num_qubits, 2)
        parameters = parameters.view(batch_size, self.num_layers, 
                                    self.num_qubits, 2)
        
        return parameters
    
    def generate_circuit_parameters(self, system_properties: np.ndarray) \
            -> np.ndarray:
        """
        Generate circuit parameters (numpy interface).
        
        Args:
            system_properties: Array of shape (num_features,)
            
        Returns:
            Circuit parameters array of shape (num_layers, num_qubits, 2)
        """
        # Convert to torch tensor
        system_props_torch = torch.FloatTensor(system_properties).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            parameters = self.forward(system_props_torch)
        
        # Convert back to numpy
        return parameters[0].numpy()
    
    def update(self, loss: torch.Tensor) -> float:
        """
        Update model weights based on loss.
        
        Args:
            loss: Scalar loss value
            
        Returns:
            Loss value as float
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return float(loss.detach())
    
    def set_learning_rate(self, lr: float):
        """
        Update the learning rate.
        
        Args:
            lr: New learning rate
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        logger.info(f"Learning rate updated to {lr}")


class ParameterizedCircuitModel:
    """
    Parameterized circuit model that generates quantum circuits.
    
    This model takes circuit parameters and constructs actual quantum circuits,
    handling the mapping between generated parameters and circuit gates.
    """
    
    def __init__(self, num_qubits: int, num_layers: int = 2,
                 entanglement: str = 'full'):
        """
        Initialize parameterized circuit model.
        
        Args:
            num_qubits: Number of qubits
            num_layers: Number of circuit layers
            entanglement: Type of entanglement ('none', 'linear', 'full')
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.entanglement = entanglement
    
    def build_circuit(self, parameters: np.ndarray, simulator):
        """
        Build quantum circuit from parameters.
        
        Args:
            parameters: Parameter array of shape (num_layers, num_qubits, 2)
            simulator: QuantumSimulator instance
            
        Returns:
            QuantumCircuit object
        """
        circuit = simulator.create_circuit(self.num_qubits)
        
        # Add parameterized layers
        for layer_idx in range(self.num_layers):
            # Get parameters for this layer
            rx_params = parameters[layer_idx, :, 0]
            ry_params = parameters[layer_idx, :, 1]
            
            # Add RX and RY rotations
            circuit.rx(rx_params, range(self.num_qubits))
            circuit.ry(ry_params, range(self.num_qubits))
            
            # Add entanglement
            if self.entanglement == 'none':
                pass
            elif self.entanglement == 'linear':
                for i in range(self.num_qubits - 1):
                    circuit.cx(i, i + 1)
            elif self.entanglement == 'full':
                # Full entanglement (all-to-all CNOT pattern)
                for i in range(self.num_qubits):
                    for j in range(i + 1, self.num_qubits):
                        circuit.cx(i, j)
        
        return circuit
    
    def get_circuit_info(self, parameters: np.ndarray) -> Dict:
        """
        Get information about the circuit without building it.
        
        Args:
            parameters: Parameter array
            
        Returns:
            Dictionary with circuit info
        """
        info = {
            'num_qubits': self.num_qubits,
            'num_layers': self.num_layers,
            'num_parameters': parameters.size,
            'entanglement': self.entanglement,
            'estimated_depth': self._estimate_depth()
        }
        return info
    
    def _estimate_depth(self) -> int:
        """Estimate circuit depth."""
        depth_per_layer = 2  # RX and RY
        if self.entanglement == 'linear':
            depth_per_layer += 1  # CNOT layer
        elif self.entanglement == 'full':
            depth_per_layer += self.num_qubits  # Many CNOT gates
        
        return self.num_layers * depth_per_layer


class AdaptiveGenerativeModel(nn.Module):
    """
    Adaptive generative model that adjusts its capacity based on system size.
    
    This model can dynamically adjust hidden layer size based on the number
    of qubits, enabling better scalability.
    """
    
    def __init__(self, num_qubits: int, num_layers: int = 2,
                 adaptive: bool = True, learning_rate: float = 0.01):
        """
        Initialize adaptive generative model.
        
        Args:
            num_qubits: Number of qubits
            num_layers: Number of circuit layers
            adaptive: If True, scale hidden dimension with num_qubits
            learning_rate: Optimizer learning rate
        """
        super().__init__()
        
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.adaptive = adaptive
        
        # Adaptive hidden dimension
        if adaptive:
            hidden_dim = max(32, num_qubits * 4)  # Scale with system size
        else:
            hidden_dim = 64
        
        input_dim = 4
        output_dim = num_layers * num_qubits * 2
        
        # Build network with adaptive sizing
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Additional hidden layers for larger systems
        if num_qubits > 8:
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        logger.info(f"Initialized AdaptiveGenerativeModel: {num_qubits} qubits, "
                   f"hidden_dim={hidden_dim}, adaptive={adaptive}")
    
    def forward(self, system_properties: torch.Tensor) -> torch.Tensor:
        """
        Generate circuit parameters.
        
        Args:
            system_properties: Input properties tensor
            
        Returns:
            Circuit parameters tensor
        """
        batch_size = system_properties.shape[0]
        output = self.network(system_properties)
        parameters = torch.sigmoid(output) * (2 * np.pi)
        parameters = parameters.view(batch_size, self.num_layers,
                                    self.num_qubits, 2)
        return parameters
    
    def update(self, loss: torch.Tensor) -> float:
        """Update model weights."""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())


def create_default_system_properties(num_qubits: int) -> np.ndarray:
    """
    Create default system properties for testing.
    
    Args:
        num_qubits: Number of qubits
        
    Returns:
        System properties array of shape (4,)
    """
    # Simplified properties: [bond_length, charge, symmetry, scale]
    properties = np.array([
        0.735,           # Bond length (Angstroms)
        0.0,             # Net charge
        1.0,             # Symmetry parameter
        num_qubits / 4   # System scale factor
    ], dtype=np.float32)
    
    return properties
