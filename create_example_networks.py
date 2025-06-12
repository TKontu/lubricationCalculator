#!/usr/bin/env python3
"""
Script to create example network configurations using the NetworkBuilder
"""

from lubrication_flow_package.utils.network_builder import (
    NetworkBuilder, create_simple_tree, create_complex_network, create_example_networks
)


def main():
    """Create various example networks"""
    print("Creating example network configurations...")
    
    # Method 1: Use predefined templates
    print("\n1. Creating predefined templates...")
    
    simple = create_simple_tree("My Simple Tree", 0.018, 220000.0)
    simple.save_json("examples/my_simple_tree.json")
    print("   ✅ Created: examples/my_simple_tree.json")
    
    complex_net = create_complex_network("My Complex Network", 0.030, 280000.0)
    complex_net.save_json("examples/my_complex_network.json")
    print("   ✅ Created: examples/my_complex_network.json")
    
    # Method 2: Build custom network step by step
    print("\n2. Building custom network step by step...")
    
    custom = (NetworkBuilder("Industrial Lubrication System", 
                           "Multi-zone industrial lubrication network")
              # Add nodes
              .add_inlet("pump_outlet", "Pump Outlet", 0.0, 0.0, 0.0)
              .add_junction("main_manifold", "Main Manifold", 1.0, 15.0, 0.0)
              .add_junction("zone1_manifold", "Zone 1 Manifold", 2.0, 25.0, 10.0)
              .add_junction("zone2_manifold", "Zone 2 Manifold", 2.0, 25.0, -10.0)
              .add_outlet("bearing1", "Bearing 1", 2.5, 35.0, 15.0)
              .add_outlet("bearing2", "Bearing 2", 2.5, 35.0, 5.0)
              .add_outlet("bearing3", "Bearing 3", 2.5, 35.0, -5.0)
              .add_outlet("bearing4", "Bearing 4", 2.5, 35.0, -15.0)
              
              # Add components
              .add_channel("main_supply", 0.12, 15.0, "Main Supply Line")
              .add_channel("zone1_supply", 0.08, 12.0, "Zone 1 Supply")
              .add_channel("zone2_supply", 0.08, 12.0, "Zone 2 Supply")
              .add_channel("bearing1_line", 0.04, 8.0, "Bearing 1 Line")
              .add_channel("bearing2_line", 0.04, 6.0, "Bearing 2 Line")
              .add_channel("bearing3_line", 0.04, 6.0, "Bearing 3 Line")
              .add_channel("bearing4_line", 0.04, 8.0, "Bearing 4 Line")
              .add_nozzle("nozzle1", 0.030, "venturi", "Bearing 1 Nozzle")
              .add_nozzle("nozzle2", 0.025, "rounded", "Bearing 2 Nozzle")
              .add_nozzle("nozzle3", 0.025, "rounded", "Bearing 3 Nozzle")
              .add_nozzle("nozzle4", 0.030, "venturi", "Bearing 4 Nozzle")
              
              # Make connections
              .connect("pump_outlet", "main_manifold", "main_supply")
              .connect("main_manifold", "zone1_manifold", "zone1_supply")
              .connect("main_manifold", "zone2_manifold", "zone2_supply")
              .connect("zone1_manifold", "bearing1", "bearing1_line")
              .connect("zone1_manifold", "bearing2", "bearing2_line")
              .connect("zone2_manifold", "bearing3", "bearing3_line")
              .connect("zone2_manifold", "bearing4", "bearing4_line")
              
              # Set simulation parameters
              .set_simulation_params(
                  total_flow_rate=0.040,  # 40 L/s
                  temperature=50.0,       # 50°C operating temperature
                  inlet_pressure=300000.0, # 300 kPa pump pressure
                  oil_type="SAE40",       # Heavier oil for industrial use
                  oil_density=920.0       # Slightly denser oil
              ))
    
    custom.save_json("examples/industrial_system.json")
    custom.save_xml("examples/industrial_system.xml")
    print("   ✅ Created: examples/industrial_system.json")
    print("   ✅ Created: examples/industrial_system.xml")
    
    # Method 3: Create a high-pressure precision system
    print("\n3. Creating high-pressure precision system...")
    
    precision = (NetworkBuilder("Precision Machining Lubrication", 
                               "High-pressure precision lubrication for CNC machines")
                .add_inlet("high_pressure_pump", "High Pressure Pump", 0.0, 0.0, 0.0)
                .add_junction("pressure_regulator", "Pressure Regulator", 0.5, 5.0, 0.0)
                .add_junction("spindle_manifold", "Spindle Manifold", 1.0, 10.0, 0.0)
                .add_junction("tool_manifold", "Tool Manifold", 1.0, 15.0, 0.0)
                .add_outlet("spindle_bearing", "Spindle Bearing", 1.2, 12.0, 3.0)
                .add_outlet("tool_coolant", "Tool Coolant", 1.0, 18.0, 0.0)
                .add_outlet("way_lubrication", "Way Lubrication", 0.8, 20.0, -3.0)
                
                .add_channel("main_line", 0.06, 5.0, "Main Pressure Line")
                .add_channel("spindle_line", 0.04, 3.0, "Spindle Line")
                .add_channel("tool_line", 0.04, 5.0, "Tool Line")
                .add_channel("way_line", 0.03, 8.0, "Way Lubrication Line")
                .add_nozzle("spindle_nozzle", 0.015, "flow_nozzle", "Precision Spindle Nozzle")
                .add_nozzle("tool_nozzle", 0.012, "venturi", "Tool Coolant Nozzle")
                .add_nozzle("way_nozzle", 0.020, "sharp_edged", "Way Lubrication Nozzle")
                
                .connect("high_pressure_pump", "pressure_regulator", "main_line")
                .connect("pressure_regulator", "spindle_manifold", "spindle_line")
                .connect("pressure_regulator", "tool_manifold", "tool_line")
                .connect("spindle_manifold", "spindle_bearing", "spindle_nozzle")
                .connect("tool_manifold", "tool_coolant", "tool_nozzle")
                .connect("tool_manifold", "way_lubrication", "way_line")
                
                .set_simulation_params(
                    total_flow_rate=0.008,   # 8 L/s - lower flow for precision
                    temperature=25.0,        # 25°C - controlled temperature
                    inlet_pressure=500000.0, # 500 kPa - high pressure system
                    oil_type="SAE10",        # Light oil for precision
                    oil_density=850.0        # Light oil density
                ))
    
    precision.save_json("examples/precision_system.json")
    print("   ✅ Created: examples/precision_system.json")
    
    print("\n🎉 All example networks created successfully!")
    print("\nTo simulate these networks, use:")
    print("   python main.py network simulate examples/my_simple_tree.json")
    print("   python main.py network simulate examples/industrial_system.json")
    print("   python main.py network simulate examples/precision_system.json")


if __name__ == "__main__":
    main()