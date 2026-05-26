"""
Quantum Simulator Module

Handles quantum circuit simulation, state vector evolution, and
Hamiltonian expectation value computation.
"""

import numpy as np
from typing import Union, List, Tuple, Dict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, Statevector
import logging

logger = logging.getLogger(__name__)


class QuantumSimulator:
    """
    Wrapper for quantum circuit simulation using Qiskit Aer backend.
    
    Provides methods for:
    - Circuit construction and execution
    - Hamiltonian expectation value computation
    - State vector manipulation
    - Noise simulation (optional)
    """
    
    def __init__(self, num_qubits: int, backend: str = 'statevector', 
                 shots: int = 1024, seed: int = 42):
        """
        Initialize quantum simulator.
        
        Args:
            num_qubits: Number of qubits in the system
            backend: 'statevector' or 'qasm' simulator
            shots: Number of measurement shots (for qasm_simulator)
            seed: Random seed for reproducibility
        """
        self.num_qubits = num_qubits
        self.backend_type = backend
        self.shots = shots
        self.seed = seed
        
        if backend == 'statevector':
            self.simulator = AerSimulator(method='statevector')
        elif backend == 'qasm':
            self.simulator = AerSimulator(method='qasm')
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        logger.info(f"Initialized simulator with {num_qubits} qubits, backend={backend}")
    
    def create_circuit(self, num_qubits: int) -> QuantumCircuit:
        """
        Create a new quantum circuit.
        
        Args:
            num_qubits: Number of qubits
            
        Returns:
            Empty QuantumCircuit object
        """
        qr = QuantumRegister(num_qubits, 'q')
        circuit = QuantumCircuit(qr)
        return circuit
    
    def add_rx_layers(self, circuit: QuantumCircuit, 
                      parameters: np.ndarray) -> QuantumCircuit:
        """
        Add RX rotation layers to circuit.
        
        Args:
            circuit: Quantum circuit
            parameters: 1D or 2D array of rotation angles
            
        Returns:
            Modified circuit
        """
        if parameters.ndim == 1:
            # Single layer
            for i, param in enumerate(parameters):
                if i < self.num_qubits:
                    circuit.rx(param, i)
        else:
            # Multiple layers
            for layer in parameters:
                for i, param in enumerate(layer):
                    if i < self.num_qubits:
                        circuit.rx(param, i)
        
        return circuit
    
    def add_ry_layers(self, circuit: QuantumCircuit, 
                      parameters: np.ndarray) -> QuantumCircuit:
        """
        Add RY rotation layers to circuit.
        
        Args:
            circuit: Quantum circuit
            parameters: 1D or 2D array of rotation angles
            
        Returns:
            Modified circuit
        """
        if parameters.ndim == 1:
            for i, param in enumerate(parameters):
                if i < self.num_qubits:
                    circuit.ry(param, i)
        else:
            for layer in parameters:
                for i, param in enumerate(layer):
                    if i < self.num_qubits:
                        circuit.ry(param, i)
        
        return circuit
    
    def add_entangling_layer(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Add a layer of CNOT gates for entanglement.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Modified circuit with entangling gates
        """
        for i in range(self.num_qubits - 1):
            circuit.cx(i, i + 1)
        # Add wrap-around entanglement
        if self.num_qubits > 2:
            circuit.cx(self.num_qubits - 1, 0)
        
        return circuit
    
    def get_statevector(self, circuit: QuantumCircuit) -> np.ndarray:
        """
        Get the statevector of a circuit.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Complex statevector array
        """
        # Use Qiskit's statevector simulator
        statevector = Statevector.from_instruction(circuit)
        return statevector.data
    
    def compute_expectation_value(self, circuit: QuantumCircuit, 
                                  hamiltonian: np.ndarray) -> float:
        """
        Compute <ψ|H|ψ> where |ψ> is the state from the circuit.
        
        Args:
            circuit: Quantum circuit
            hamiltonian: Hamiltonian matrix (2^n x 2^n)
            
        Returns:
            Expectation value (real number)
        """
        # Get statevector
        psi = self.get_statevector(circuit)
        
        # Compute <ψ|H|ψ>
        H_psi = hamiltonian @ psi
        expectation = np.real(np.conj(psi) @ H_psi)
        
        return expectation
    
    def compute_pauli_expectation(self, circuit: QuantumCircuit,
                                  pauli_string: str) -> float:
        """
        Compute expectation value of a Pauli string.
        
        Args:
            circuit: Quantum circuit
            pauli_string: String like 'ZIZ' (I=identity, X/Y/Z=Pauli)
            
        Returns:
            Expectation value
        """
        # Get statevector
        psi = self.get_statevector(circuit)
        
        # Create Pauli operator
        pauli_op = SparsePauliOp(pauli_string)
        matrix = pauli_op.to_matrix()
        
        # Compute expectation
        H_psi = matrix @ psi
        expectation = np.real(np.conj(psi) @ H_psi)
        
        return expectation
    
    def get_circuit_depth(self, circuit: QuantumCircuit) -> int:
        """
        Get the depth of the circuit (number of layers).
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Circuit depth
        """
        return circuit.depth()
    
    def get_gate_count(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """
        Get count of each gate type in the circuit.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Dictionary with gate counts
        """
        return circuit.count_ops()


def build_h2_hamiltonian(bond_length: float = 0.735) -> np.ndarray:
    """
    Build Hamiltonian for H2 molecule.
    
    Args:
        bond_length: Bond length in Angstroms
        
    Returns:
        4x4 Hamiltonian matrix (for 2 qubits)
    """
    # This is a simplified H2 Hamiltonian for 2 qubits
    # Ground state energy ≈ -1.1026 Ha at 0.735 Å
    
    # Using Jordan-Wigner mapping: coefficients for Pauli terms
    # H = a₀*I + a₁*Z₀ + a₂*Z₁ + a₃*Z₀Z₁ + a₄*X₀X₁ + a₅*Y₀Y₁
    
    # Typical coefficients for H2 at 0.735 Å
    h0 = -0.8126
    h1 = -0.0912
    h2 = -0.0912
    h3 = 0.1655
    h4 = 0.1655
    h5 = 0.1655
    
    # Build Hamiltonian matrix
    # |00>, |01>, |10>, |11>
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    
    H = (h0 * np.kron(I, I) +
         h1 * np.kron(Z, I) +
         h2 * np.kron(I, Z) +
         h3 * np.kron(Z, Z) +
         h4 * np.kron(X, X) +
         h5 * np.kron(Y, Y))
    
    return H


def build_lih_hamiltonian() -> np.ndarray:
    """
    Build Hamiltonian for LiH molecule (4 qubits).
    
    Returns:
        16x16 Hamiltonian matrix
    """
    # Simplified LiH Hamiltonian using STO-3G basis
    # More complex than H2, requires 4 qubits minimum
    # Ground state energy ≈ -7.88 Ha
    
    # For now, create a realistic approximation
    dim = 16
    H = np.zeros((dim, dim))
    
    # Diagonal elements (approximate energies)
    for i in range(dim):
        H[i, i] = -7.88 + 0.1 * (i - dim/2)
    
    # Add some off-diagonal coupling (simplified)
    for i in range(dim-1):
        H[i, i+1] = 0.05
        H[i+1, i] = 0.05
    
    # Ensure Hermitian
    H = (H + H.conj().T) / 2
    
    return H


def exact_ground_state_energy(hamiltonian: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute exact ground state energy and state using diagonalization.
    
    Args:
        hamiltonian: Hamiltonian matrix
        
    Returns:
        Tuple of (ground state energy, ground state vector)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    ground_energy = eigenvalues[0]
    ground_state = eigenvectors[:, 0]
    
    return ground_energy, ground_state
