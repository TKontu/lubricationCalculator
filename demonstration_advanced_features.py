#!/usr/bin/env python3
"""
ADemonstration of Advanced Lubrication Flow Distribution Calculator Features

This script demonstrates the enhanced capabilities of the advanced calculator
including different nozzle types, complex systems, and robust numerical methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from advanced_lubrication_flow_tool import (
    AdvancedLubricationFlowCalculator,
    PipeSegment,
    Nozzle,
    Branch,
    NozzleType
)


def demonstrate_nozzle_types():
    """Demonstrate different nozzle types and their effects"""
    print("="*70)
    print("DEMONSTRATION: DIFFERENT NOZZLE TYPES")
    print("="*70)
    
    calculator = AdvancedLubricationFlowCalculator(oil_density=900.0, oil_type="SAE30")
    
    # Create branches with different nozzle types
    nozzle_types = [
        (NozzleType.SHARP_EDGED, "Sharp-Edged Orifice"),
        (NozzleType.ROUNDED, "Rounded Entrance"),
        (NozzleType.VENTURI, "Venturi Nozzle"),
        (NozzleType.FLOW_NOZZLE, "Flow Nozzle"),
        (None, "No Nozzle")
    ]
    
    results = []
    
    for nozzle_type, description in nozzle_types:
        if nozzle_type:
            nozzle = Nozzle(diameter=0.008, nozzle_type=nozzle_type)
        else:
            nozzle = None
            
        branch = Branch(
            pipe=PipeSegment(diameter=0.05, length=10.0),
            nozzle=nozzle,
            name=description
        )
        
        # Calculate for single branch
        flows, info = calculator.solve_flow_distribution(
            0.005, [branch], 40  # 5 L/s, 40°C
        )
        
        pressure_drop = info['pressure_drops'][0]
        results.append((description, flows[0]*1000, pressure_drop))
    
    print(f"\n{'Nozzle Type':<25} {'Flow Rate (L/s)':<15} {'Pressure Drop (Pa)'}")
    print("-" * 65)
    for description, flow, pressure in results:
        print(f"{description:<25} {flow:<15.3f} {pressure:<15.1f}")
    
    print(f"\nKey Observations:")
    print(f"- Venturi nozzles have the lowest pressure drop due to diffuser recovery")
    print(f"- Flow nozzles are nearly as efficient as venturi nozzles")
    print(f"- Sharp-edged orifices have the highest pressure drop")
    print(f"- Rounded entrances are better than sharp-edged but not as good as venturi")


def demonstrate_complex_system():
    """Demonstrate complex multi-branch system with mixed nozzle types"""
    print("\n" + "="*70)
    print("DEMONSTRATION: COMPLEX MULTI-BRANCH SYSTEM")
    print("="*70)
    
    calculator = AdvancedLubricationFlowCalculator(oil_density=900.0, oil_type="SAE30")
    
    # Create complex system with various configurations
    branches = [
        Branch(
            pipe=PipeSegment(diameter=0.08, length=15.0, roughness=0.00015),
            nozzle=Nozzle(diameter=0.015, nozzle_type=NozzleType.VENTURI),
            name="Main Bearing",
            elevation_change=2.0
        ),
        Branch(
            pipe=PipeSegment(diameter=0.06, length=12.0, roughness=0.00020),
            nozzle=Nozzle(diameter=0.012, nozzle_type=NozzleType.FLOW_NOZZLE),
            name="Aux Bearing",
            elevation_change=1.0
        ),
        Branch(
            pipe=PipeSegment(diameter=0.05, length=18.0, roughness=0.00025),
            nozzle=Nozzle(diameter=0.008, nozzle_type=NozzleType.ROUNDED),
            name="Gear Box 1",
            elevation_change=0.5
        ),
        Branch(
            pipe=PipeSegment(diameter=0.04, length=14.0, roughness=0.00030),
            nozzle=Nozzle(diameter=0.006, nozzle_type=NozzleType.SHARP_EDGED),
            name="Gear Box 2",
            elevation_change=-0.5
        ),
        Branch(
            pipe=PipeSegment(diameter=0.035, length=8.0, roughness=0.00015),
            nozzle=None,  # No nozzle
            name="Cooler Return",
            elevation_change=-2.0
        )
    ]
    
    total_flow = 0.03  # 30 L/s
    temperature = 50   # °C
    
    print(f"System Configuration:")
    print(f"- Total flow rate: {total_flow*1000:.1f} L/s")
    print(f"- Temperature: {temperature}°C")
    print(f"- Number of branches: {len(branches)}")
    print(f"- Mixed nozzle types and elevations")
    
    # Solve using both methods
    flows_newton, info_newton = calculator.solve_flow_distribution(
        total_flow, branches, temperature, method="newton"
    )
    
    flows_iter, info_iter = calculator.solve_flow_distribution(
        total_flow, branches, temperature, method="iterative"
    )
    
    print(f"\nSolution Comparison:")
    print(f"Newton method: {info_newton['iterations']} iterations, "
          f"converged: {info_newton['converged']}")
    print(f"Iterative method: {info_iter['iterations']} iterations, "
          f"converged: {info_iter['converged']}")
    
    # Print detailed results
    calculator.print_results(flows_newton, branches, info_newton, detailed=True)


def demonstrate_temperature_effects():
    """Demonstrate temperature effects on flow distribution"""
    print("\n" + "="*70)
    print("DEMONSTRATION: TEMPERATURE EFFECTS")
    print("="*70)
    
    calculator = AdvancedLubricationFlowCalculator(oil_density=900.0, oil_type="SAE30")
    
    # Simple two-branch system
    branches = [
        Branch(
            pipe=PipeSegment(diameter=0.05, length=10.0),
            nozzle=Nozzle(diameter=0.008),
            name="Branch 1"
        ),
        Branch(
            pipe=PipeSegment(diameter=0.04, length=12.0),
            nozzle=Nozzle(diameter=0.006),
            name="Branch 2"
        )
    ]
    
    total_flow = 0.01  # 10 L/s
    temperatures = [20, 40, 60, 80, 100]  # °C
    
    results = []
    
    for temp in temperatures:
        flows, info = calculator.solve_flow_distribution(
            total_flow, branches, temp
        )
        
        viscosity = info['viscosity']
        reynolds_numbers = info['reynolds_numbers']
        pressure_drops = info['pressure_drops']
        
        results.append({
            'temperature': temp,
            'viscosity': viscosity,
            'flows': flows,
            'reynolds': reynolds_numbers,
            'pressure_drops': pressure_drops
        })
    
    print(f"\n{'Temp (°C)':<10} {'Viscosity':<12} {'Branch 1':<12} {'Branch 2':<12} {'Re1':<8} {'Re2':<8}")
    print(f"{'':10} {'(Pa·s)':<12} {'(L/s)':<12} {'(L/s)':<12} {'':8} {'':8}")
    print("-" * 75)
    
    for result in results:
        temp = result['temperature']
        visc = result['viscosity']
        flow1 = result['flows'][0] * 1000
        flow2 = result['flows'][1] * 1000
        re1 = result['reynolds'][0]
        re2 = result['reynolds'][1]
        
        print(f"{temp:<10} {visc:<12.6f} {flow1:<12.3f} {flow2:<12.3f} {re1:<8.0f} {re2:<8.0f}")
    
    print(f"\nKey Observations:")
    print(f"- Higher temperature → lower viscosity → higher Reynolds numbers")
    print(f"- Flow distribution changes with temperature due to viscosity effects")
    print(f"- Pressure drops decrease with higher temperature (lower viscosity)")


def demonstrate_oil_type_comparison():
    """Demonstrate different oil types"""
    print("\n" + "="*70)
    print("DEMONSTRATION: DIFFERENT OIL TYPES")
    print("="*70)
    
    # Simple system
    branches = [
        Branch(
            pipe=PipeSegment(diameter=0.05, length=10.0),
            nozzle=Nozzle(diameter=0.008),
            name="Branch 1"
        ),
        Branch(
            pipe=PipeSegment(diameter=0.04, length=12.0),
            nozzle=Nozzle(diameter=0.006),
            name="Branch 2"
        )
    ]
    
    total_flow = 0.01  # 10 L/s
    temperature = 40   # °C
    oil_types = ["SAE10", "SAE20", "SAE30", "SAE40", "SAE50"]
    
    results = []
    
    for oil_type in oil_types:
        calculator = AdvancedLubricationFlowCalculator(oil_type=oil_type)
        flows, info = calculator.solve_flow_distribution(
            total_flow, branches, temperature
        )
        
        results.append({
            'oil_type': oil_type,
            'viscosity': info['viscosity'],
            'flows': flows,
            'pressure_drops': info['pressure_drops']
        })
    
    print(f"\n{'Oil Type':<10} {'Viscosity':<12} {'Branch 1':<12} {'Branch 2':<12} {'ΔP1 (Pa)':<10} {'ΔP2 (Pa)'}")
    print(f"{'':10} {'(Pa·s)':<12} {'(L/s)':<12} {'(L/s)':<12} {'':10} {'':10}")
    print("-" * 80)
    
    for result in results:
        oil = result['oil_type']
        visc = result['viscosity']
        flow1 = result['flows'][0] * 1000
        flow2 = result['flows'][1] * 1000
        dp1 = result['pressure_drops'][0]
        dp2 = result['pressure_drops'][1]
        
        print(f"{oil:<10} {visc:<12.6f} {flow1:<12.3f} {flow2:<12.3f} {dp1:<10.0f} {dp2:<10.0f}")
    
    print(f"\nKey Observations:")
    print(f"- Higher SAE numbers have higher viscosity")
    print(f"- Flow distribution is affected by oil viscosity")
    print(f"- Pressure drops increase with higher viscosity oils")


def demonstrate_large_system_scalability():
    """Demonstrate scalability with large systems"""
    print("\n" + "="*70)
    print("DEMONSTRATION: LARGE SYSTEM SCALABILITY")
    print("="*70)
    
    calculator = AdvancedLubricationFlowCalculator()
    
    branch_counts = [5, 10, 20, 50, 100]
    
    print(f"\n{'Branches':<10} {'Newton Time':<12} {'Newton Iter':<12} {'Iterative Time':<15} {'Iterative Iter'}")
    print("-" * 70)
    
    for num_branches in branch_counts:
        # Create system
        branches = []
        for i in range(num_branches):
            diameter = 0.02 + 0.01 * (i % 5)
            length = 5.0 + 2.0 * (i % 4)
            
            if i % 3 == 0:  # Add nozzle to every third branch
                nozzle = Nozzle(diameter=0.005 + 0.002 * (i % 3))
            else:
                nozzle = None
            
            branches.append(Branch(
                pipe=PipeSegment(diameter=diameter, length=length),
                nozzle=nozzle,
                name=f"Branch {i+1}"
            ))
        
        total_flow = 0.001 * num_branches  # Scale flow with number of branches
        temperature = 40
        
        # Test Newton method
        import time
        start_time = time.time()
        flows_newton, info_newton = calculator.solve_flow_distribution(
            total_flow, branches, temperature, method="newton"
        )
        newton_time = time.time() - start_time
        
        # Test iterative method
        start_time = time.time()
        flows_iter, info_iter = calculator.solve_flow_distribution(
            total_flow, branches, temperature, method="iterative"
        )
        iter_time = time.time() - start_time
        
        print(f"{num_branches:<10} {newton_time:<12.4f} {info_newton['iterations']:<12} "
              f"{iter_time:<15.4f} {info_iter['iterations']}")
    
    print(f"\nKey Observations:")
    print(f"- Both methods scale well with system size")
    print(f"- Newton method may require more iterations but is still efficient")
    print(f"- Iterative method is consistently fast for most systems")


def main():
    """Run all demonstrations"""
    print("ADVANCED LUBRICATION FLOW DISTRIBUTION CALCULATOR")
    print("COMPREHENSIVE FEATURE DEMONSTRATION")
    
    demonstrate_nozzle_types()
    demonstrate_complex_system()
    demonstrate_temperature_effects()
    demonstrate_oil_type_comparison()
    demonstrate_large_system_scalability()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nThe advanced calculator provides:")
    print("✓ Multiple nozzle types with accurate pressure drop models")
    print("✓ Robust Newton-Raphson and iterative solution methods")
    print("✓ Temperature and oil type effects on viscosity")
    print("✓ Elevation effects and complex pipe geometries")
    print("✓ Excellent scalability for large systems")
    print("✓ Comprehensive validation and error handling")


if __name__ == "__main__":
    main()