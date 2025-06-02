#!/usr/bin/env python3
"""
ALubrication Flow Distribution GUI Application

A comprehensive graphical user interface for the lubrication piping flow 
distribution calculation tool. Features include:
- Interactive branch configuration
- Real-time flow distribution visualization
- Temperature and fluid property controls
- Export capabilities for results
- System optimization tools
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json
from improved_lubrication_flow_tool import (
    LubricationFlowCalculator, Branch, PipeSegment, Nozzle
)


class LubricationFlowGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lubrication Flow Distribution Calculator")
        self.root.geometry("1200x800")
        
        # Initialize calculator
        self.calculator = LubricationFlowCalculator()
        
        # Data storage
        self.branches = []
        self.results = None
        
        # Create GUI
        self.create_widgets()
        self.load_default_system()
        
    def create_widgets(self):
        """Create the main GUI widgets"""
        
        # Create main frames
        self.create_control_frame()
        self.create_results_frame()
        self.create_plot_frame()
        
    def create_control_frame(self):
        """Create the control panel frame"""
        
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # System parameters
        sys_frame = ttk.LabelFrame(control_frame, text="System Parameters")
        sys_frame.pack(fill=tk.X, pady=5)
        
        # Total flow rate
        ttk.Label(sys_frame, text="Total Flow Rate (L/s):").grid(row=0, column=0, sticky=tk.W)
        self.total_flow_var = tk.DoubleVar(value=10.0)
        ttk.Entry(sys_frame, textvariable=self.total_flow_var, width=10).grid(row=0, column=1)
        
        # Temperature
        ttk.Label(sys_frame, text="Temperature (°C):").grid(row=1, column=0, sticky=tk.W)
        self.temperature_var = tk.DoubleVar(value=40.0)
        ttk.Entry(sys_frame, textvariable=self.temperature_var, width=10).grid(row=1, column=1)
        
        # Oil type
        ttk.Label(sys_frame, text="Oil Type:").grid(row=2, column=0, sticky=tk.W)
        self.oil_type_var = tk.StringVar(value="SAE30")
        oil_combo = ttk.Combobox(sys_frame, textvariable=self.oil_type_var, 
                                values=["SAE10", "SAE30", "SAE50"], width=8)
        oil_combo.grid(row=2, column=1)
        
        # Oil density
        ttk.Label(sys_frame, text="Oil Density (kg/m³):").grid(row=3, column=0, sticky=tk.W)
        self.oil_density_var = tk.DoubleVar(value=900.0)
        ttk.Entry(sys_frame, textvariable=self.oil_density_var, width=10).grid(row=3, column=1)
        
        # Branches frame
        branches_frame = ttk.LabelFrame(control_frame, text="Branches Configuration")
        branches_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Branches treeview
        self.branches_tree = ttk.Treeview(branches_frame, columns=('Pipe D', 'Pipe L', 'Nozzle D'), height=8)
        self.branches_tree.heading('#0', text='Branch Name')
        self.branches_tree.heading('Pipe D', text='Pipe D (mm)')
        self.branches_tree.heading('Pipe L', text='Pipe L (m)')
        self.branches_tree.heading('Nozzle D', text='Nozzle D (mm)')
        
        self.branches_tree.column('#0', width=100)
        self.branches_tree.column('Pipe D', width=80)
        self.branches_tree.column('Pipe L', width=80)
        self.branches_tree.column('Nozzle D', width=80)
        
        self.branches_tree.pack(fill=tk.BOTH, expand=True)
        
        # Branch control buttons
        btn_frame = ttk.Frame(branches_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Add Branch", command=self.add_branch).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Edit Branch", command=self.edit_branch).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Delete Branch", command=self.delete_branch).pack(side=tk.LEFT, padx=2)
        
        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, text="Calculate", command=self.calculate_flow, 
                  style="Accent.TButton").pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Save System", command=self.save_system).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Load System", command=self.load_system).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Export Results", command=self.export_results).pack(fill=tk.X, pady=2)
        
    def create_results_frame(self):
        """Create the results display frame"""
        
        results_frame = ttk.LabelFrame(self.root, text="Results")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results text widget with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(text_frame, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_plot_frame(self):
        """Create the plotting frame"""
        
        plot_frame = ttk.LabelFrame(self.root, text="Flow Distribution Visualization")
        plot_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def add_branch(self):
        """Add a new branch"""
        BranchDialog(self.root, self.on_branch_added)
        
    def edit_branch(self):
        """Edit selected branch"""
        selection = self.branches_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a branch to edit")
            return
            
        item = selection[0]
        index = self.branches_tree.index(item)
        branch = self.branches[index]
        
        BranchDialog(self.root, self.on_branch_edited, branch, index)
        
    def delete_branch(self):
        """Delete selected branch"""
        selection = self.branches_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a branch to delete")
            return
            
        if messagebox.askyesno("Confirm", "Are you sure you want to delete this branch?"):
            item = selection[0]
            index = self.branches_tree.index(item)
            del self.branches[index]
            self.update_branches_display()
            
    def on_branch_added(self, branch):
        """Callback when a branch is added"""
        self.branches.append(branch)
        self.update_branches_display()
        
    def on_branch_edited(self, branch, index):
        """Callback when a branch is edited"""
        self.branches[index] = branch
        self.update_branches_display()
        
    def update_branches_display(self):
        """Update the branches treeview"""
        # Clear existing items
        for item in self.branches_tree.get_children():
            self.branches_tree.delete(item)
            
        # Add branches
        for branch in self.branches:
            pipe_d = branch.pipe.diameter * 1000  # Convert to mm
            pipe_l = branch.pipe.length
            nozzle_d = branch.nozzle.diameter * 1000 if branch.nozzle else "None"
            
            self.branches_tree.insert('', 'end', text=branch.name,
                                    values=(f"{pipe_d:.1f}", f"{pipe_l:.1f}", 
                                           f"{nozzle_d:.1f}" if nozzle_d != "None" else "None"))
    
    def calculate_flow(self):
        """Calculate flow distribution"""
        if not self.branches:
            messagebox.showwarning("Warning", "Please add at least one branch")
            return
            
        try:
            # Update calculator properties
            self.calculator.oil_density = self.oil_density_var.get()
            self.calculator.oil_type = self.oil_type_var.get()
            
            # Calculate flow distribution
            total_flow = self.total_flow_var.get() / 1000  # Convert L/s to m³/s
            temperature = self.temperature_var.get()
            
            branch_flows, solution_info = self.calculator.solve_flow_distribution(
                total_flow, self.branches, temperature
            )
            
            self.results = {
                'branch_flows': branch_flows,
                'solution_info': solution_info,
                'total_flow': total_flow,
                'temperature': temperature
            }
            
            # Display results
            self.display_results()
            self.plot_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {str(e)}")
            
    def display_results(self):
        """Display calculation results in text widget"""
        if not self.results:
            return
            
        self.results_text.delete(1.0, tk.END)
        
        # Format results similar to the console output
        text = "LUBRICATION FLOW DISTRIBUTION RESULTS\n"
        text += "=" * 50 + "\n\n"
        
        info = self.results['solution_info']
        text += f"Temperature: {info['temperature']:.1f}°C\n"
        text += f"Oil Type: {self.calculator.oil_type}\n"
        text += f"Oil Density: {self.calculator.oil_density:.1f} kg/m³\n"
        text += f"Dynamic Viscosity: {info['viscosity']:.6f} Pa·s\n"
        text += f"Converged: {info['converged']} (in {info['iterations']} iterations)\n\n"
        
        text += f"{'Branch':<15} {'Flow Rate':<12} {'Pressure Drop':<15} {'Reynolds':<10}\n"
        text += f"{'Name':<15} {'(L/s)':<12} {'(Pa)':<15} {'Number':<10}\n"
        text += "-" * 55 + "\n"
        
        total_flow = 0
        for i, (flow, branch) in enumerate(zip(self.results['branch_flows'], self.branches)):
            flow_lps = flow * 1000
            pressure_drop = info['pressure_drops'][i]
            reynolds = info['reynolds_numbers'][i]
            
            text += f"{branch.name:<15} {flow_lps:<12.3f} {pressure_drop:<15.1f} {reynolds:<10.0f}\n"
            total_flow += flow
            
        text += "-" * 55 + "\n"
        text += f"{'Total':<15} {total_flow*1000:<12.3f}\n"
        
        self.results_text.insert(1.0, text)
        
    def plot_results(self):
        """Plot flow distribution results"""
        if not self.results:
            return
            
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        branch_names = [branch.name for branch in self.branches]
        flow_rates = [flow * 1000 for flow in self.results['branch_flows']]  # Convert to L/s
        pressure_drops = self.results['solution_info']['pressure_drops']
        
        # Flow rate bar chart
        bars1 = self.ax1.bar(branch_names, flow_rates, color='skyblue', alpha=0.7)
        self.ax1.set_ylabel('Flow Rate (L/s)')
        self.ax1.set_title('Flow Distribution by Branch')
        self.ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, flow_rates):
            height = bar.get_height()
            self.ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{value:.2f}', ha='center', va='bottom')
        
        # Pressure drop bar chart
        bars2 = self.ax2.bar(branch_names, pressure_drops, color='lightcoral', alpha=0.7)
        self.ax2.set_ylabel('Pressure Drop (Pa)')
        self.ax2.set_title('Pressure Drop by Branch')
        self.ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, pressure_drops):
            height = bar.get_height()
            self.ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                         f'{value:.0f}', ha='center', va='bottom')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def save_system(self):
        """Save current system configuration"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                system_data = {
                    'total_flow': self.total_flow_var.get(),
                    'temperature': self.temperature_var.get(),
                    'oil_type': self.oil_type_var.get(),
                    'oil_density': self.oil_density_var.get(),
                    'branches': []
                }
                
                for branch in self.branches:
                    branch_data = {
                        'name': branch.name,
                        'pipe_diameter': branch.pipe.diameter,
                        'pipe_length': branch.pipe.length,
                        'pipe_roughness': branch.pipe.roughness,
                        'nozzle_diameter': branch.nozzle.diameter if branch.nozzle else None,
                        'nozzle_discharge_coeff': branch.nozzle.discharge_coeff if branch.nozzle else None
                    }
                    system_data['branches'].append(branch_data)
                
                with open(filename, 'w') as f:
                    json.dump(system_data, f, indent=2)
                    
                messagebox.showinfo("Success", "System saved successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save system: {str(e)}")
                
    def load_system(self):
        """Load system configuration"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    system_data = json.load(f)
                
                # Load system parameters
                self.total_flow_var.set(system_data['total_flow'])
                self.temperature_var.set(system_data['temperature'])
                self.oil_type_var.set(system_data['oil_type'])
                self.oil_density_var.set(system_data['oil_density'])
                
                # Load branches
                self.branches = []
                for branch_data in system_data['branches']:
                    pipe = PipeSegment(
                        diameter=branch_data['pipe_diameter'],
                        length=branch_data['pipe_length'],
                        roughness=branch_data['pipe_roughness']
                    )
                    
                    nozzle = None
                    if branch_data['nozzle_diameter']:
                        nozzle = Nozzle(
                            diameter=branch_data['nozzle_diameter'],
                            discharge_coeff=branch_data['nozzle_discharge_coeff']
                        )
                    
                    branch = Branch(pipe=pipe, nozzle=nozzle, name=branch_data['name'])
                    self.branches.append(branch)
                
                self.update_branches_display()
                messagebox.showinfo("Success", "System loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load system: {str(e)}")
                
    def export_results(self):
        """Export calculation results"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export. Please calculate first.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("Success", "Results exported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
                
    def load_default_system(self):
        """Load a default system for demonstration"""
        self.branches = [
            Branch(
                pipe=PipeSegment(diameter=0.05, length=5.0),
                nozzle=Nozzle(diameter=0.008, discharge_coeff=0.6),
                name="Main Bearing"
            ),
            Branch(
                pipe=PipeSegment(diameter=0.04, length=6.0),
                nozzle=Nozzle(diameter=0.006, discharge_coeff=0.6),
                name="Aux Bearing"
            ),
            Branch(
                pipe=PipeSegment(diameter=0.03, length=7.0),
                nozzle=Nozzle(diameter=0.004, discharge_coeff=0.6),
                name="Gear Box"
            ),
            Branch(
                pipe=PipeSegment(diameter=0.025, length=4.0),
                nozzle=None,
                name="Cooler Return"
            )
        ]
        self.update_branches_display()


class BranchDialog:
    def __init__(self, parent, callback, branch=None, index=None):
        self.callback = callback
        self.branch = branch
        self.index = index
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Branch Configuration")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_widgets()
        
        if branch:
            self.load_branch_data()
            
    def create_widgets(self):
        """Create dialog widgets"""
        
        # Branch name
        ttk.Label(self.dialog, text="Branch Name:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(self.dialog, textvariable=self.name_var, width=20).grid(row=0, column=1, padx=10, pady=5)
        
        # Pipe parameters
        pipe_frame = ttk.LabelFrame(self.dialog, text="Pipe Parameters")
        pipe_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, padx=10, pady=10)
        
        ttk.Label(pipe_frame, text="Diameter (mm):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.pipe_diameter_var = tk.DoubleVar()
        ttk.Entry(pipe_frame, textvariable=self.pipe_diameter_var, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(pipe_frame, text="Length (m):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.pipe_length_var = tk.DoubleVar()
        ttk.Entry(pipe_frame, textvariable=self.pipe_length_var, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(pipe_frame, text="Roughness (mm):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.pipe_roughness_var = tk.DoubleVar(value=0.15)
        ttk.Entry(pipe_frame, textvariable=self.pipe_roughness_var, width=15).grid(row=2, column=1, padx=5, pady=2)
        
        # Nozzle parameters
        nozzle_frame = ttk.LabelFrame(self.dialog, text="Nozzle Parameters")
        nozzle_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=10, pady=10)
        
        self.has_nozzle_var = tk.BooleanVar()
        ttk.Checkbutton(nozzle_frame, text="Has Nozzle", variable=self.has_nozzle_var,
                       command=self.toggle_nozzle).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(nozzle_frame, text="Diameter (mm):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.nozzle_diameter_var = tk.DoubleVar()
        self.nozzle_diameter_entry = ttk.Entry(nozzle_frame, textvariable=self.nozzle_diameter_var, width=15)
        self.nozzle_diameter_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(nozzle_frame, text="Discharge Coeff:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.nozzle_coeff_var = tk.DoubleVar(value=0.6)
        self.nozzle_coeff_entry = ttk.Entry(nozzle_frame, textvariable=self.nozzle_coeff_var, width=15)
        self.nozzle_coeff_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="OK", command=self.ok_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        self.toggle_nozzle()
        
    def toggle_nozzle(self):
        """Enable/disable nozzle parameters based on checkbox"""
        state = tk.NORMAL if self.has_nozzle_var.get() else tk.DISABLED
        self.nozzle_diameter_entry.config(state=state)
        self.nozzle_coeff_entry.config(state=state)
        
    def load_branch_data(self):
        """Load existing branch data into dialog"""
        self.name_var.set(self.branch.name)
        self.pipe_diameter_var.set(self.branch.pipe.diameter * 1000)  # Convert to mm
        self.pipe_length_var.set(self.branch.pipe.length)
        self.pipe_roughness_var.set(self.branch.pipe.roughness * 1000)  # Convert to mm
        
        if self.branch.nozzle:
            self.has_nozzle_var.set(True)
            self.nozzle_diameter_var.set(self.branch.nozzle.diameter * 1000)  # Convert to mm
            self.nozzle_coeff_var.set(self.branch.nozzle.discharge_coeff)
        else:
            self.has_nozzle_var.set(False)
            
        self.toggle_nozzle()
        
    def ok_clicked(self):
        """Handle OK button click"""
        try:
            # Create pipe segment
            pipe = PipeSegment(
                diameter=self.pipe_diameter_var.get() / 1000,  # Convert to meters
                length=self.pipe_length_var.get(),
                roughness=self.pipe_roughness_var.get() / 1000  # Convert to meters
            )
            
            # Create nozzle if specified
            nozzle = None
            if self.has_nozzle_var.get():
                nozzle = Nozzle(
                    diameter=self.nozzle_diameter_var.get() / 1000,  # Convert to meters
                    discharge_coeff=self.nozzle_coeff_var.get()
                )
            
            # Create branch
            branch = Branch(pipe=pipe, nozzle=nozzle, name=self.name_var.get())
            
            # Call callback
            if self.index is not None:
                self.callback(branch, self.index)
            else:
                self.callback(branch)
                
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")


def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = LubricationFlowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()