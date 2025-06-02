#!/usr/bin/env python3
"""
ABenchmark and Validation Tool for Lubrication Flow Distribution Calculators

This tool provides comprehensive benchmarking and validation of the lubrication
flow distribution calculators with various system configurations.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json

from advanced_lubrication_flow_tool import (
    AdvancedLubricationFlowCalculator,
    PipeSegment,
    Nozzle,
    Branch,
    NozzleType
)

from improved_lubrication_flow_tool import (
    LubricationFlowCalculator as BasicCalculator,
    PipeSegment as BasicPipeSegment,
    Nozzle as BasicNozzle,
    Branch as BasicBranch
)


class BenchmarkSuite:
    """Comprehensive benchmark suite for lubrication flow calculators"""
    
    def __init__(self):
        self.advanced_calc = AdvancedLubricationFlowCalculator(oil_density=900.0, oil_type="SAE30")
        self.basic_calc = BasicCalculator(oil_density=900.0, oil_type="SAE30")
        self.results = {}
    
    def create_simple_system(self) -> Tuple[float, List, List, float]:
        """Create simple 2-branch system for both calculators"""
        total_flow = 0.01  # m³/s
        temperature = 40   # °C
        
        # Advanced calculator branches
        advanced_branches = [
            Branch(
                pipe=PipeSegment(diameter=0.05, length=5.0),
                nozzle=Nozzle(diameter=0.008, nozzle_type=NozzleType.SHARP_EDGED),
                name="Branch 1"
            ),
            Branch(
                pipe=PipeSegment(diameter=0.04, length=6.0),
                nozzle=Nozzle(diameter=0.006, nozzle_type=NozzleType.SHARP_EDGED),
                name="Branch 2"
            )
        ]
        
        # Basic calculator branches (compatible format)
        basic_branches = [
            BasicBranch(
                pipe=BasicPipeSegment(diameter=0.05, length=5.0),
                nozzle=BasicNozzle(diameter=0.008, discharge_coeff=0.6),
                name="Branch 1"
            ),
            BasicBranch(
                pipe=BasicPipeSegment(diameter=0.04, length=6.0),
                nozzle=BasicNozzle(diameter=0.006, discharge_coeff=0.6),
                name="Branch 2"
            )
        ]
        
        return total_flow, advanced_branches, basic_branches, temperature
    
    def create_complex_system(self) -> Tuple[float, List, List, float]:
        """Create complex multi-branch system"""
        total_flow = 0.025  # m³/s
        temperature = 45    # °C
        
        # System configuration
        configs = [
            {"d_pipe": 0.08, "l_pipe": 12.0, "d_nozzle": 0.015, "nozzle_type": NozzleType.SHARP_EDGED},
            {"d_pipe": 0.06, "l_pipe": 8.0, "d_nozzle": 0.012, "nozzle_type": NozzleType.ROUNDED},
            {"d_pipe": 0.05, "l_pipe": 15.0, "d_nozzle": 0.008, "nozzle_type": NozzleType.VENTURI},
            {"d_pipe": 0.04, "l_pipe": 10.0, "d_nozzle": 0.006, "nozzle_type": NozzleType.FLOW_NOZZLE},
            {"d_pipe": 0.035, "l_pipe": 18.0, "d_nozzle": 0.005, "nozzle_type": NozzleType.SHARP_EDGED},
            {"d_pipe": 0.03, "l_pipe": 14.0, "d_nozzle": None, "nozzle_type": None},  # No nozzle
        ]
        
        advanced_branches = []
        basic_branches = []
        
        for i, config in enumerate(configs):
            # Advanced branch
            if config["d_nozzle"]:
                nozzle = Nozzle(diameter=config["d_nozzle"], nozzle_type=config["nozzle_type"])
                basic_nozzle = BasicNozzle(diameter=config["d_nozzle"], discharge_coeff=nozzle.discharge_coeff)
            else:
                nozzle = None
                basic_nozzle = None
            
            advanced_branches.append(Branch(
                pipe=PipeSegment(diameter=config["d_pipe"], length=config["l_pipe"]),
                nozzle=nozzle,
                name=f"Branch {i+1}"
            ))
            
            basic_branches.append(BasicBranch(
                pipe=BasicPipeSegment(diameter=config["d_pipe"], length=config["l_pipe"]),
                nozzle=basic_nozzle,
                name=f"Branch {i+1}"
            ))
        
        return total_flow, advanced_branches, basic_branches, temperature
    
    def create_large_system(self, num_branches: int = 20) -> Tuple[float, List, List, float]:
        """Create large system with many branches"""
        total_flow = 0.05  # m³/s
        temperature = 40   # °C
        
        advanced_branches = []
        basic_branches = []
        
        for i in range(num_branches):
            # Vary parameters to create realistic diversity
            diameter = 0.02 + 0.01 * (i % 5)
            length = 5.0 + 2.0 * (i % 4)
            
            # Add nozzle to every other branch
            if i % 2 == 0:
                nozzle_diameter = 0.003 + 0.002 * (i % 3)
                nozzle = Nozzle(diameter=nozzle_diameter, nozzle_type=NozzleType.SHARP_EDGED)
                basic_nozzle = BasicNozzle(diameter=nozzle_diameter, discharge_coeff=0.6)
            else:
                nozzle = None
                basic_nozzle = None
            
            advanced_branches.append(Branch(
                pipe=PipeSegment(diameter=diameter, length=length),
                nozzle=nozzle,
                name=f"Branch {i+1}"
            ))
            
            basic_branches.append(BasicBranch(
                pipe=BasicPipeSegment(diameter=diameter, length=length),
                nozzle=basic_nozzle,
                name=f"Branch {i+1}"
            ))
        
        return total_flow, advanced_branches, basic_branches, temperature
    
    def benchmark_system(self, name: str, total_flow: float, advanced_branches: List, 
                        basic_branches: List, temperature: float) -> Dict:
        """Benchmark a system configuration"""
        print(f"\nBenchmarking {name}...")
        print(f"Branches: {len(advanced_branches)}, Flow: {total_flow*1000:.1f} L/s, Temp: {temperature}°C")
        
        results = {
            'name': name,
            'num_branches': len(advanced_branches),
            'total_flow': total_flow,
            'temperature': temperature
        }
        
        # Test Advanced Calculator - Newton method
        start_time = time.time()
        try:
            flows_adv_newton, info_adv_newton = self.advanced_calc.solve_flow_distribution(
                total_flow, advanced_branches, temperature, method="newton"
            )
            time_adv_newton = time.time() - start_time
            results['advanced_newton'] = {
                'success': True,
                'time': time_adv_newton,
                'iterations': info_adv_newton['iterations'],
                'converged': info_adv_newton['converged'],
                'flows': flows_adv_newton,
                'pressure_drops': info_adv_newton['pressure_drops']
            }
        except Exception as e:
            results['advanced_newton'] = {'success': False, 'error': str(e)}
        
        # Test Advanced Calculator - Iterative method
        start_time = time.time()
        try:
            flows_adv_iter, info_adv_iter = self.advanced_calc.solve_flow_distribution(
                total_flow, advanced_branches, temperature, method="iterative"
            )
            time_adv_iter = time.time() - start_time
            results['advanced_iterative'] = {
                'success': True,
                'time': time_adv_iter,
                'iterations': info_adv_iter['iterations'],
                'converged': info_adv_iter['converged'],
                'flows': flows_adv_iter,
                'pressure_drops': info_adv_iter['pressure_drops']
            }
        except Exception as e:
            results['advanced_iterative'] = {'success': False, 'error': str(e)}
        
        # Test Basic Calculator
        start_time = time.time()
        try:
            flows_basic, info_basic = self.basic_calc.solve_flow_distribution(
                total_flow, basic_branches, temperature
            )
            time_basic = time.time() - start_time
            results['basic'] = {
                'success': True,
                'time': time_basic,
                'iterations': info_basic['iterations'],
                'converged': info_basic['converged'],
                'flows': flows_basic,
                'pressure_drops': info_basic['pressure_drops']
            }
        except Exception as e:
            results['basic'] = {'success': False, 'error': str(e)}
        
        # Print summary
        print(f"Results:")
        for method in ['advanced_newton', 'advanced_iterative', 'basic']:
            if results[method]['success']:
                r = results[method]
                print(f"  {method:18}: {r['time']:.4f}s, {r['iterations']:2d} iter, "
                      f"converged: {r['converged']}")
            else:
                print(f"  {method:18}: FAILED - {results[method]['error']}")
        
        return results
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite"""
        print("="*70)
        print("COMPREHENSIVE LUBRICATION FLOW CALCULATOR BENCHMARK")
        print("="*70)
        
        # Simple system
        total_flow, adv_branches, basic_branches, temp = self.create_simple_system()
        self.results['simple'] = self.benchmark_system(
            "Simple System (2 branches)", total_flow, adv_branches, basic_branches, temp
        )
        
        # Complex system
        total_flow, adv_branches, basic_branches, temp = self.create_complex_system()
        self.results['complex'] = self.benchmark_system(
            "Complex System (6 branches)", total_flow, adv_branches, basic_branches, temp
        )
        
        # Large systems
        for num_branches in [10, 20, 50]:
            total_flow, adv_branches, basic_branches, temp = self.create_large_system(num_branches)
            self.results[f'large_{num_branches}'] = self.benchmark_system(
                f"Large System ({num_branches} branches)", 
                total_flow, adv_branches, basic_branches, temp
            )
    
    def analyze_accuracy(self):
        """Analyze accuracy differences between methods"""
        print(f"\n{'='*70}")
        print("ACCURACY ANALYSIS")
        print(f"{'='*70}")
        
        for system_name, result in self.results.items():
            if not all(result[method]['success'] for method in ['advanced_newton', 'advanced_iterative', 'basic']):
                continue
                
            print(f"\n{system_name.upper()}:")
            
            flows_newton = np.array(result['advanced_newton']['flows'])
            flows_iter = np.array(result['advanced_iterative']['flows'])
            flows_basic = np.array(result['basic']['flows'])
            
            # Compare flows
            diff_newton_iter = np.max(np.abs(flows_newton - flows_iter)) / np.max(flows_newton) * 100
            diff_newton_basic = np.max(np.abs(flows_newton - flows_basic)) / np.max(flows_newton) * 100
            diff_iter_basic = np.max(np.abs(flows_iter - flows_basic)) / np.max(flows_iter) * 100
            
            print(f"  Max flow difference (Newton vs Iterative): {diff_newton_iter:.3f}%")
            print(f"  Max flow difference (Newton vs Basic):     {diff_newton_basic:.3f}%")
            print(f"  Max flow difference (Iterative vs Basic):  {diff_iter_basic:.3f}%")
            
            # Mass conservation check
            mass_error_newton = abs(sum(flows_newton) - result['total_flow']) / result['total_flow'] * 100
            mass_error_iter = abs(sum(flows_iter) - result['total_flow']) / result['total_flow'] * 100
            mass_error_basic = abs(sum(flows_basic) - result['total_flow']) / result['total_flow'] * 100
            
            print(f"  Mass conservation error (Newton):    {mass_error_newton:.6f}%")
            print(f"  Mass conservation error (Iterative): {mass_error_iter:.6f}%")
            print(f"  Mass conservation error (Basic):     {mass_error_basic:.6f}%")
    
    def analyze_performance(self):
        """Analyze performance characteristics"""
        print(f"\n{'='*70}")
        print("PERFORMANCE ANALYSIS")
        print(f"{'='*70}")
        
        # Collect performance data
        branch_counts = []
        newton_times = []
        iter_times = []
        basic_times = []
        newton_iters = []
        iter_iters = []
        basic_iters = []
        
        for system_name, result in self.results.items():
            if not all(result[method]['success'] for method in ['advanced_newton', 'advanced_iterative', 'basic']):
                continue
                
            branch_counts.append(result['num_branches'])
            newton_times.append(result['advanced_newton']['time'])
            iter_times.append(result['advanced_iterative']['time'])
            basic_times.append(result['basic']['time'])
            newton_iters.append(result['advanced_newton']['iterations'])
            iter_iters.append(result['advanced_iterative']['iterations'])
            basic_iters.append(result['basic']['iterations'])
        
        # Print performance summary
        print(f"\n{'System':<20} {'Branches':<10} {'Newton':<15} {'Iterative':<15} {'Basic':<15}")
        print(f"{'Name':<20} {'Count':<10} {'Time(s)/Iter':<15} {'Time(s)/Iter':<15} {'Time(s)/Iter':<15}")
        print("-" * 80)
        
        for i, (name, result) in enumerate(self.results.items()):
            if not all(result[method]['success'] for method in ['advanced_newton', 'advanced_iterative', 'basic']):
                continue
                
            newton_str = f"{newton_times[i]:.4f}/{newton_iters[i]}"
            iter_str = f"{iter_times[i]:.4f}/{iter_iters[i]}"
            basic_str = f"{basic_times[i]:.4f}/{basic_iters[i]}"
            
            print(f"{name:<20} {branch_counts[i]:<10} {newton_str:<15} {iter_str:<15} {basic_str:<15}")
        
        # Performance insights
        print(f"\nPERFORMANCE INSIGHTS:")
        if len(newton_times) > 0:
            avg_newton_time = np.mean(newton_times)
            avg_iter_time = np.mean(iter_times)
            avg_basic_time = np.mean(basic_times)
            
            print(f"  Average solution time:")
            print(f"    Newton method:    {avg_newton_time:.4f}s")
            print(f"    Iterative method: {avg_iter_time:.4f}s")
            print(f"    Basic method:     {avg_basic_time:.4f}s")
            
            avg_newton_iter = np.mean(newton_iters)
            avg_iter_iter = np.mean(iter_iters)
            avg_basic_iter = np.mean(basic_iters)
            
            print(f"  Average iterations:")
            print(f"    Newton method:    {avg_newton_iter:.1f}")
            print(f"    Iterative method: {avg_iter_iter:.1f}")
            print(f"    Basic method:     {avg_basic_iter:.1f}")
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for system_name, result in self.results.items():
            json_results[system_name] = {}
            for method, method_result in result.items():
                if isinstance(method_result, dict) and 'flows' in method_result:
                    json_results[system_name][method] = method_result.copy()
                    if isinstance(method_result['flows'], np.ndarray):
                        json_results[system_name][method]['flows'] = method_result['flows'].tolist()
                    if isinstance(method_result['pressure_drops'], np.ndarray):
                        json_results[system_name][method]['pressure_drops'] = method_result['pressure_drops'].tolist()
                else:
                    json_results[system_name][method] = method_result
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {filename}")


def main():
    """Run comprehensive benchmark suite"""
    benchmark = BenchmarkSuite()
    
    # Run all benchmarks
    benchmark.run_comprehensive_benchmark()
    
    # Analyze results
    benchmark.analyze_accuracy()
    benchmark.analyze_performance()
    
    # Save results
    benchmark.save_results()
    
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()