import tkinter as tk
from tkinter import ttk, simpledialog

class PropertiesEditor(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.entries = {}

    def show_properties(self, element_id, properties):
        for widget in self.winfo_children():
            widget.destroy()
        self.entries.clear()

        ttk.Label(self, text=f"Properties for {element_id}").pack(pady=5)

        for key, value in properties.items():
            frame = ttk.Frame(self)
            frame.pack(fill=tk.X, padx=5, pady=2)
            label = ttk.Label(frame, text=f"{key}:")
            label.pack(side=tk.LEFT)
            if key == 'type':
                entry = ttk.Combobox(frame, values=["Node", "inlet", "outlet"])
                entry.set(value)
            else:
                entry = ttk.Entry(frame)
                entry.insert(0, str(value))
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            self.entries[key] = entry

        save_button = ttk.Button(self, text="Save", command=lambda: self.save_properties(element_id))
        save_button.pack(pady=5)

    def save_properties(self, element_id):
        new_properties = {key: entry.get() for key, entry in self.entries.items()}
        self.app.update_element_properties(element_id, new_properties)

class ComponentDialog(simpledialog.Dialog):
    def __init__(self, parent, title, component_type, properties):
        self.component_type = component_type
        self.properties = properties
        super().__init__(parent, title=title)

    def body(self, master):
        self.entries = {}
        for key, value in self.properties.items():
            ttk.Label(master, text=f"{key}:").grid(row=len(self.entries), sticky=tk.W)
            entry = ttk.Entry(master)
            entry.insert(0, str(value))
            entry.grid(row=len(self.entries), column=1, padx=5, pady=5)
            self.entries[key] = entry
        return self.entries[list(self.properties.keys())[0]]

    def apply(self):
        self.result = {key: float(entry.get()) for key, entry in self.entries.items()}
