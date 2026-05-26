"""
GETTING STARTED WITH GQE PROTOTYPE
===================================

This script provides a quick-start guide to the GQE implementation.
Run this to verify everything is working correctly.
"""

import sys
import os

def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def check_dependencies():
    """Check if all required packages are installed."""
    print_header("CHECKING DEPENDENCIES")
    
    required_packages = [
        'numpy', 'scipy', 'torch', 'qiskit', 'qiskit_aer',
        'matplotlib', 'pandas', 'sklearn', 'tqdm'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package:<20} OK")
        except ImportError:
            print(f"✗ {package:<20} MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nTo install, run:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def run_quick_test():
    """Run a quick test to verify the implementation works."""
    print_header("RUNNING QUICK TEST (H2 molecule)")
    
    try:
        from gqe_prototype import GenerativeQuantumEigensolver
        from quantum_simulator import build_h2_hamiltonian, exact_ground_state_energy
        
        # Load H2 Hamiltonian
        H2_hamiltonian = build_h2_hamiltonian()
        exact_energy, _ = exact_ground_state_energy(H2_hamiltonian)
        
        print(f"H2 Exact Ground State Energy: {exact_energy:.6f} Ha")
        print(f"Target: Chemical accuracy < 1.6 mHa\n")
        
        # Create and run GQE
        print("Initializing GQE solver...")
        gqe = GenerativeQuantumEigensolver(
            num_qubits=2,
            num_layers=2,
            learning_rate=0.01,
            verbose=True
        )
        
        print("\nRunning optimization (20 iterations for quick test)...\n")
        result = gqe.optimize(
            hamiltonian=H2_hamiltonian,
            max_iterations=20,
            target_accuracy=1.6e-3
        )
        
        print("\n" + "-" * 70)
        print("TEST RESULTS")
        print("-" * 70)
        print(f"GQE Energy:        {result['energy']:.6f} Ha")
        print(f"Exact Energy:      {result['exact_energy']:.6f} Ha")
        print(f"Absolute Error:    {result['error']:.6f} Ha")
        print(f"Error (mHa):       {result['error_mha']:.4f} mHa")
        print(f"Circuit Depth:     {result['circuit_depth']}")
        print(f"Iterations:        {result['iterations']}")
        
        if result['error_mha'] < 1.6:
            print(f"\n✓ SUCCESS! Chemical accuracy achieved!")
        else:
            print(f"\n⚠ Note: Target not reached with 20 iterations.")
            print(f"  Try increasing max_iterations for better results.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def show_usage_examples():
    """Show common usage examples."""
    print_header("COMMON USAGE EXAMPLES")
    
    examples = {
        "1. Basic GQE on H2": """
from gqe_prototype import GenerativeQuantumEigensolver
from quantum_simulator import build_h2_hamiltonian

H2_hamiltonian = build_h2_hamiltonian()
gqe = GenerativeQuantumEigensolver(num_qubits=2, num_layers=2)
result = gqe.optimize(hamiltonian=H2_hamiltonian, max_iterations=100)
print(f"Energy: {result['energy']:.6f} Ha")
print(f"Error: {result['error_mha']:.4f} mHa")
        """,
        
        "2. Run Scalability Study": """
python examples/scalability_study.py
        """,
        
        "3. Compare GQE vs VQE": """
from gqe_prototype import GenerativeQuantumEigensolver, VQEBaseline
from benchmark import BenchmarkSuite

benchmark = BenchmarkSuite()
result = benchmark.benchmark_gqe_vs_vqe(
    hamiltonian=hamiltonian,
    num_qubits=4,
    num_trials=3
)
        """,
        
        "4. Custom Hamiltonian": """
import numpy as np
from gqe_prototype import GenerativeQuantumEigensolver

# Create custom Hamiltonian
dim = 16  # 2^4 for 4 qubits
H = np.random.randn(dim, dim)
H = (H + H.conj().T) / 2  # Make Hermitian

gqe = GenerativeQuantumEigensolver(num_qubits=4)
result = gqe.optimize(hamiltonian=H, max_iterations=100)
        """,
        
        "5. Analyze Convergence": """
from gqe_prototype import GenerativeQuantumEigensolver
from benchmark import BenchmarkSuite

gqe = GenerativeQuantumEigensolver(num_qubits=2)
result = gqe.optimize(hamiltonian=H2_hamiltonian, max_iterations=200)

benchmark = BenchmarkSuite()
benchmark.plot_convergence(result['history'], save_path='convergence.png')
        """
    }
    
    for title, code in examples.items():
        print(f"\n{title}")
        print("-" * 70)
        print(code.strip())


def show_next_steps():
    """Show next steps for challenge participants."""
    print_header("NEXT STEPS FOR CHALLENGE")
    
    steps = """
1. UNDERSTAND THE PROTOTYPE
   □ Read README.md in gqe_prototype/
   □ Run h2_ground_state.py to see basic example
   □ Review code structure and comments

2. EXPERIMENT WITH HYPERPARAMETERS
   □ Try different learning rates (0.001 to 0.1)
   □ Vary circuit depth (1-4 layers)
   □ Test different entanglement types
   □ Use scalability_study.py to test scaling

3. IMPLEMENT YOUR APPROACH
   □ Design your target molecular system
   □ Create custom Hamiltonian (use PySCF, OpenFermion, etc.)
   □ Adapt generative model for your system
   □ Implement optimization improvements

4. BENCHMARK & OPTIMIZE
   □ Compare against VQE baseline
   □ Measure accuracy, speed, circuit depth
   □ Document scaling behavior
   □ Identify and address bottlenecks

5. PREPARE SUBMISSION
   □ Write 3-page technical description
   □ Prepare code with clear run instructions
   □ Include benchmarking results
   □ Provide reproducibility documentation

KEY FOCUS AREAS FOR SUBMISSION:
- Scalability (primary criterion)
- Accuracy (< 1.6 mHa where possible)
- Algorithmic innovation
- Computational efficiency
- Clear hybrid architecture
- Benchmarking strategy

Remember: Quality over quantity!
- Focus on demonstrating one approach well
- Show clear understanding of limitations
- Provide detailed analysis and insights
- Make results reproducible
    """
    
    print(steps)


def main():
    """Main entry point."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║   GQE PROTOTYPE - GETTING STARTED GUIDE                             ║
║   Generative Quantum Eigensolver Implementation                      ║
║   Global Industry Challenge 2026 - Phase 2                          ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠ Please install missing packages first.")
        return False
    
    # Run quick test
    if not run_quick_test():
        print("\n⚠ Quick test failed. Check error messages above.")
        return False
    
    # Show examples
    show_usage_examples()
    
    # Show next steps
    show_next_steps()
    
    # Summary
    print_header("QUICK REFERENCE")
    
    reference = """
PROJECT STRUCTURE:
├── gqe_prototype.py           Main GQE implementation
├── quantum_simulator.py       Quantum circuit simulation
├── generative_model.py        Neural network models
├── benchmark.py               Benchmarking tools
├── requirements.txt           Python dependencies
├── examples/
│   ├── h2_ground_state.py    Basic example
│   ├── scalability_study.py  Scaling analysis
│   └── advanced_customization.py Advanced features
└── results/                   Output directory

IMPORTANT FILES:
- Read: README.md (main documentation)
- Study: gqe_prototype.py (core algorithm)
- Run: python examples/h2_ground_state.py
- Extend: examples/advanced_customization.py

KEY RESOURCES:
- Challenge PDF: Contains all requirements
- README.md: Complete guide and requirements
- Code Comments: Detailed explanations
- Examples: Working reference implementations

GETTING HELP:
1. Check code comments and docstrings
2. Review examples for usage patterns
3. Refer to challenge PDF for requirements
4. Search for function/class documentation

Remember: This is a starting template. You need to:
- Understand the algorithm deeply
- Develop your own innovations
- Test thoroughly on your target systems
- Document results comprehensively
    """
    
    print(reference)
    
    print_header("READY TO GET STARTED!")
    print("""
Next recommended steps:

1. Run the basic example:
   python examples/h2_ground_state.py

2. Modify the example for your target molecule:
   - Create new file in examples/
   - Load your custom Hamiltonian
   - Run GQE and compare results

3. Study the advanced example:
   python examples/advanced_customization.py

4. Implement your scalability improvements:
   - Edit generative_model.py
   - Add optimization techniques
   - Test on larger systems

Good luck with the challenge!
    """)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
