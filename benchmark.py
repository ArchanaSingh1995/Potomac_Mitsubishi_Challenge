"""
Benchmarking Module

Tools for comparing GQE with VQE and classical methods,
analyzing scalability, and generating performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time
import logging

from gqe_prototype import GenerativeQuantumEigensolver, VQEBaseline
from quantum_simulator import (QuantumSimulator, exact_ground_state_energy,
                               build_h2_hamiltonian, build_lih_hamiltonian)

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for GQE vs VQE and other methods.
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize benchmark suite."""
        self.verbose = verbose
        self.results = {}
    
    def benchmark_gqe_vs_vqe(self, hamiltonian: np.ndarray,
                             num_qubits: int,
                             num_trials: int = 3) -> Dict:
        """
        Compare GQE and VQE on the same problem.
        
        Args:
            hamiltonian: Target Hamiltonian
            num_qubits: Number of qubits
            num_trials: Number of independent trials
            
        Returns:
            Comparison results dictionary
        """
        results = {
            'gqe': {'energies': [], 'errors': [], 'times': []},
            'vqe': {'energies': [], 'errors': [], 'times': []}
        }
        
        exact_energy, _ = exact_ground_state_energy(hamiltonian)
        
        # GQE trials
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running GQE trials ({num_trials} trials)")
            print(f"{'='*60}")
        
        for trial in range(num_trials):
            if self.verbose:
                print(f"\nGQE Trial {trial+1}/{num_trials}")
            
            gqe = GenerativeQuantumEigensolver(
                num_qubits=num_qubits,
                num_layers=2,
                learning_rate=0.01,
                verbose=self.verbose
            )
            
            start_time = time.time()
            gqe_result = gqe.optimize(
                hamiltonian=hamiltonian,
                max_iterations=100,
                target_accuracy=1.6e-3
            )
            elapsed_time = time.time() - start_time
            
            results['gqe']['energies'].append(gqe_result['energy'])
            results['gqe']['errors'].append(gqe_result['error_mha'])
            results['gqe']['times'].append(elapsed_time)
        
        # VQE trials
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running VQE trials ({num_trials} trials)")
            print(f"{'='*60}")
        
        for trial in range(num_trials):
            if self.verbose:
                print(f"\nVQE Trial {trial+1}/{num_trials}")
            
            vqe = VQEBaseline(num_qubits=num_qubits, num_layers=2)
            
            start_time = time.time()
            vqe_result = vqe.optimize(
                hamiltonian=hamiltonian,
                max_iterations=100
            )
            elapsed_time = time.time() - start_time
            
            results['vqe']['energies'].append(vqe_result['energy'])
            results['vqe']['errors'].append(vqe_result['error_mha'])
            results['vqe']['times'].append(elapsed_time)
        
        # Compute statistics
        comparison = {
            'exact_energy': exact_energy,
            'gqe': {
                'mean_energy': np.mean(results['gqe']['energies']),
                'std_energy': np.std(results['gqe']['energies']),
                'mean_error_mha': np.mean(results['gqe']['errors']),
                'std_error_mha': np.std(results['gqe']['errors']),
                'mean_time': np.mean(results['gqe']['times']),
                'std_time': np.std(results['gqe']['times'])
            },
            'vqe': {
                'mean_energy': np.mean(results['vqe']['energies']),
                'std_energy': np.std(results['vqe']['energies']),
                'mean_error_mha': np.mean(results['vqe']['errors']),
                'std_error_mha': np.std(results['vqe']['errors']),
                'mean_time': np.mean(results['vqe']['times']),
                'std_time': np.std(results['vqe']['times'])
            },
            'raw_results': results
        }
        
        if self.verbose:
            self._print_comparison(comparison)
        
        return comparison
    
    def scalability_study(self, qubit_range: List[int],
                          num_trials: int = 2) -> Dict:
        """
        Study GQE scalability across different system sizes.
        
        Args:
            qubit_range: List of qubit counts to test
            num_trials: Trials per configuration
            
        Returns:
            Scalability results
        """
        results = {
            'qubits': [],
            'energy_error': [],
            'circuit_depth': [],
            'time': [],
            'gates': []
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Scalability Study: {qubit_range}")
            print(f"{'='*60}\n")
        
        for num_qubits in qubit_range:
            if self.verbose:
                print(f"Testing with {num_qubits} qubits...")
            
            # Create simple test Hamiltonian
            # For larger systems, use random symmetric matrix
            dim = 2 ** num_qubits
            H = np.random.randn(dim, dim)
            H = (H + H.conj().T) / 2  # Make Hermitian
            
            errors = []
            times = []
            depths = []
            gates = []
            
            for trial in range(num_trials):
                gqe = GenerativeQuantumEigensolver(
                    num_qubits=num_qubits,
                    num_layers=2,
                    verbose=False
                )
                
                start_time = time.time()
                result = gqe.optimize(
                    hamiltonian=H,
                    max_iterations=50,
                    target_accuracy=1.6e-3
                )
                elapsed_time = time.time() - start_time
                
                errors.append(result['error_mha'])
                times.append(elapsed_time)
                depths.append(result['circuit_depth'])
                gates.append(len(result['circuit_gates']))
            
            results['qubits'].append(num_qubits)
            results['energy_error'].append(np.mean(errors))
            results['circuit_depth'].append(np.mean(depths))
            results['time'].append(np.mean(times))
            results['gates'].append(np.mean(gates))
            
            if self.verbose:
                print(f"  Error: {np.mean(errors):.4f} ± {np.std(errors):.4f} mHa")
                print(f"  Time: {np.mean(times):.2f} ± {np.std(times):.2f} s")
                print(f"  Depth: {np.mean(depths):.1f} ± {np.std(depths):.1f}\n")
        
        return results
    
    def convergence_analysis(self, hamiltonian: np.ndarray,
                            num_qubits: int) -> Dict:
        """
        Analyze convergence behavior of GQE.
        
        Args:
            hamiltonian: Target Hamiltonian
            num_qubits: Number of qubits
            
        Returns:
            Convergence analysis results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Convergence Analysis")
            print(f"{'='*60}\n")
        
        gqe = GenerativeQuantumEigensolver(
            num_qubits=num_qubits,
            verbose=self.verbose
        )
        
        result = gqe.optimize(
            hamiltonian=hamiltonian,
            max_iterations=200
        )
        
        history = result['history']
        
        analysis = {
            'iterations_to_convergence': len(history['energy']),
            'final_energy': history['energy'][-1],
            'final_error_mha': history['error'][-1],
            'min_error_mha': min(history['error']),
            'energy_history': history['energy'],
            'error_history': history['error']
        }
        
        return analysis
    
    def _print_comparison(self, comparison: Dict):
        """Print formatted comparison results."""
        print(f"\n{'='*60}")
        print(f"GQE vs VQE Comparison Results")
        print(f"{'='*60}")
        print(f"\nExact Ground State Energy: {comparison['exact_energy']:.6f} Ha")
        
        print(f"\n{'Metric':<25} {'GQE':<25} {'VQE':<25}")
        print("-" * 75)
        
        for metric in ['mean_energy', 'mean_error_mha', 'mean_time']:
            gqe_val = comparison['gqe'].get(metric, 0)
            vqe_val = comparison['vqe'].get(metric, 0)
            
            if 'energy' in metric:
                print(f"{metric:<25} {gqe_val:>10.6f} Ha        {vqe_val:>10.6f} Ha")
            elif 'error' in metric:
                print(f"{metric:<25} {gqe_val:>10.4f} mHa       {vqe_val:>10.4f} mHa")
            else:
                print(f"{metric:<25} {gqe_val:>10.2f} s         {vqe_val:>10.2f} s")
        
        print("=" * 75)
    
    def plot_convergence(self, history: Dict, save_path: str = None):
        """
        Plot convergence history.
        
        Args:
            history: Convergence history dictionary
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Energy convergence
        axes[0].plot(history['energy'], 'b-', linewidth=2, label='Energy')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Energy (Ha)')
        axes[0].set_title('Energy Convergence')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Error convergence (log scale)
        axes[1].semilogy(history['error'], 'r-', linewidth=2, label='Absolute Error')
        axes[1].axhline(y=1.6, color='g', linestyle='--', label='Chemical Accuracy')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Error (mHa)')
        axes[1].set_title('Error Convergence')
        axes[1].grid(True, alpha=0.3, which='both')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        return fig
    
    def plot_scalability(self, results: Dict, save_path: str = None):
        """
        Plot scalability results.
        
        Args:
            results: Scalability study results
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        qubits = results['qubits']
        
        # Error vs qubits
        axes[0, 0].plot(qubits, results['energy_error'], 'bo-', linewidth=2)
        axes[0, 0].set_xlabel('Number of Qubits')
        axes[0, 0].set_ylabel('Energy Error (mHa)')
        axes[0, 0].set_title('Accuracy vs System Size')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Time vs qubits
        axes[0, 1].semilogy(qubits, results['time'], 'go-', linewidth=2)
        axes[0, 1].set_xlabel('Number of Qubits')
        axes[0, 1].set_ylabel('Time (seconds, log scale)')
        axes[0, 1].set_title('Runtime vs System Size')
        axes[0, 1].grid(True, alpha=0.3, which='both')
        
        # Circuit depth vs qubits
        axes[1, 0].plot(qubits, results['circuit_depth'], 'ro-', linewidth=2)
        axes[1, 0].set_xlabel('Number of Qubits')
        axes[1, 0].set_ylabel('Circuit Depth')
        axes[1, 0].set_title('Circuit Depth vs System Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gate count vs qubits
        axes[1, 1].plot(qubits, results['gates'], 'mo-', linewidth=2)
        axes[1, 1].set_xlabel('Number of Qubits')
        axes[1, 1].set_ylabel('Gate Count')
        axes[1, 1].set_title('Gate Count vs System Size')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        return fig


def run_quick_benchmark():
    """Quick benchmark on H2 molecule."""
    print("=" * 60)
    print("Quick Benchmark: GQE on H2 Molecule")
    print("=" * 60)
    
    # Get H2 Hamiltonian
    H2_hamiltonian = build_h2_hamiltonian()
    
    # Run GQE
    gqe = GenerativeQuantumEigensolver(
        num_qubits=2,
        num_layers=2,
        learning_rate=0.01,
        verbose=True
    )
    
    print("\nRunning GQE optimization...")
    result = gqe.optimize(
        hamiltonian=H2_hamiltonian,
        max_iterations=100,
        target_accuracy=1.6e-3
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final Energy:      {result['energy']:.6f} Ha")
    print(f"Exact Energy:      {result['exact_energy']:.6f} Ha")
    print(f"Absolute Error:    {result['error']:.6f} Ha")
    print(f"Error (mHa):       {result['error_mha']:.4f} mHa")
    print(f"Chemical Accuracy: {result['chemical_accuracy']}")
    print(f"Circuit Depth:     {result['circuit_depth']}")
    print(f"Iterations:        {result['iterations']}")
    
    return result


if __name__ == "__main__":
    # Run quick benchmark
    result = run_quick_benchmark()
