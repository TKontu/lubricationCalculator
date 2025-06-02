#!/usr/bin/env python3
"""
AWeb-based Lubrication Flow Distribution Calculator

A modern web application for lubrication piping flow distribution calculations.
Built with Flask and provides an interactive interface for:
- System configuration
- Real-time calculations
- Visualization of results
- Export capabilities
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from improved_lubrication_flow_tool import (
    LubricationFlowCalculator, Branch, PipeSegment, Nozzle
)

app = Flask(__name__)

# Global calculator instance
calculator = LubricationFlowCalculator()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/calculate', methods=['POST'])
def calculate_flow():
    """API endpoint for flow calculation"""
    try:
        data = request.json
        
        # Update calculator properties
        calculator.oil_density = data['oil_density']
        calculator.oil_type = data['oil_type']
        
        # Parse branches
        branches = []
        for branch_data in data['branches']:
            pipe = PipeSegment(
                diameter=branch_data['pipe_diameter'] / 1000,  # Convert mm to m
                length=branch_data['pipe_length'],
                roughness=branch_data['pipe_roughness'] / 1000  # Convert mm to m
            )
            
            nozzle = None
            if branch_data.get('has_nozzle') and branch_data.get('nozzle_diameter'):
                nozzle = Nozzle(
                    diameter=branch_data['nozzle_diameter'] / 1000,  # Convert mm to m
                    discharge_coeff=branch_data['nozzle_discharge_coeff']
                )
            
            branch = Branch(
                pipe=pipe,
                nozzle=nozzle,
                name=branch_data['name']
            )
            branches.append(branch)
        
        # Calculate flow distribution
        total_flow = data['total_flow'] / 1000  # Convert L/s to m³/s
        temperature = data['temperature']
        
        branch_flows, solution_info = calculator.solve_flow_distribution(
            total_flow, branches, temperature
        )
        
        # Prepare response
        results = {
            'success': True,
            'branch_flows': [flow * 1000 for flow in branch_flows],  # Convert to L/s
            'solution_info': solution_info,
            'total_flow': total_flow * 1000,  # Convert to L/s
            'temperature': temperature,
            'branches': [{'name': branch.name} for branch in branches]
        }
        
        # Generate plots
        plots = generate_plots(results)
        results['plots'] = plots
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/export', methods=['POST'])
def export_results():
    """Export results to text file"""
    try:
        data = request.json
        
        # Format results text
        text = format_results_text(data)
        
        # Create file-like object
        output = io.StringIO()
        output.write(text)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/plain',
            as_attachment=True,
            download_name='lubrication_flow_results.txt'
        )
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_plots(results):
    """Generate plots for the results"""
    try:
        branch_names = [branch['name'] for branch in results['branches']]
        flow_rates = results['branch_flows']
        pressure_drops = results['solution_info']['pressure_drops']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Flow rate bar chart
        bars1 = ax1.bar(branch_names, flow_rates, color='skyblue', alpha=0.7)
        ax1.set_ylabel('Flow Rate (L/s)')
        ax1.set_title('Flow Distribution by Branch')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, flow_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Pressure drop bar chart
        bars2 = ax2.bar(branch_names, pressure_drops, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('Pressure Drop (Pa)')
        ax2.set_title('Pressure Drop by Branch')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, pressure_drops):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Convert to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_str
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        return None

def format_results_text(data):
    """Format results as text for export"""
    text = "LUBRICATION FLOW DISTRIBUTION RESULTS\n"
    text += "=" * 50 + "\n\n"
    
    info = data['solution_info']
    text += f"Temperature: {info['temperature']:.1f}°C\n"
    text += f"Oil Type: {calculator.oil_type}\n"
    text += f"Oil Density: {calculator.oil_density:.1f} kg/m³\n"
    text += f"Dynamic Viscosity: {info['viscosity']:.6f} Pa·s\n"
    text += f"Converged: {info['converged']} (in {info['iterations']} iterations)\n\n"
    
    text += f"{'Branch':<15} {'Flow Rate':<12} {'Pressure Drop':<15} {'Reynolds':<10}\n"
    text += f"{'Name':<15} {'(L/s)':<12} {'(Pa)':<15} {'Number':<10}\n"
    text += "-" * 55 + "\n"
    
    total_flow = 0
    for i, branch in enumerate(data['branches']):
        flow_lps = data['branch_flows'][i]
        pressure_drop = info['pressure_drops'][i]
        reynolds = info['reynolds_numbers'][i]
        
        text += f"{branch['name']:<15} {flow_lps:<12.3f} {pressure_drop:<15.1f} {reynolds:<10.0f}\n"
        total_flow += flow_lps
        
    text += "-" * 55 + "\n"
    text += f"{'Total':<15} {total_flow:<12.3f}\n"
    
    return text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)