"""
Dynamic dialogs for editing properties in the GUI.
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox, QLineEdit, 
    QComboBox, QLabel
)

class PropertiesDialog(QDialog):
    """
    A dynamic dialog for editing a dictionary of properties.
    """
    def __init__(self, element_id, properties, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Properties for {element_id}")
        
        self.layout = QVBoxLayout(self)
        
        self.form_layout = QFormLayout()
        self.entries = {}

        for key, value in properties.items():
            # Make keys more readable for labels
            label_text = key.replace('_', ' ').title()
            
            if key == 'type':
                # Use a dropdown for 'type' if it's a node
                # In the future, this could be expanded for component types
                entry = QComboBox()
                entry.addItems(['internal', 'inlet', 'outlet'])
                entry.setCurrentText(str(value))
            else:
                entry = QLineEdit(str(value))
            
            self.entries[key] = entry
            self.form_layout.addRow(QLabel(label_text), entry)

        self.layout.addLayout(self.form_layout)
        
        # Standard dialog buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        self.layout.addWidget(self.button_box)

    def get_properties(self):
        """
        Returns the updated properties from the dialog entries.
        It attempts to convert values to float where possible.
        """
        updated_properties = {}
        for key, entry in self.entries.items():
            if isinstance(entry, QComboBox):
                value = entry.currentText()
            else:
                value = entry.text()

            try:
                # Attempt to convert to float, otherwise keep as string
                updated_properties[key] = float(value)
            except ValueError:
                updated_properties[key] = value
                
        return updated_properties