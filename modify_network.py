
import json

# Load the network data
with open('C:/code/lubrication3/complex_network.json', 'r') as f:
    data = json.load(f)

# Define the paths to be modified
paths_to_modify = {
    "path_4b": "1bcb5c7a",
    "path_5a": "830bcdfc",
    "path_5b": "92085884",
    "path_6": "c5c52cfb"
}

# Find and remove the old path endings
nodes_to_remove = set()
components_to_remove = set()
connections_to_remove = []

for prefix, start_node in paths_to_modify.items():
    # Find the connection that starts the section to be removed
    for conn in data['connections']:
        if conn['from_node'] == start_node:
            # Found the start of the section to remove
            current_node = conn['to_node']
            nodes_to_remove.add(current_node)
            components_to_remove.add(conn['component'])
            connections_to_remove.append(conn)

            # Follow the path to the outlet
            while True:
                found_next = False
                for next_conn in data['connections']:
                    if next_conn['from_node'] == current_node:
                        current_node = next_conn['to_node']
                        nodes_to_remove.add(current_node)
                        components_to_remove.add(next_conn['component'])
                        connections_to_remove.append(next_conn)
                        if data['nodes'] and any(n['id'] == current_node and n.get('type') == 'outlet' for n in data['nodes']):
                            found_next = True
                            break
                if not found_next:
                    break
            break


# Filter out the old nodes, components, and connections
data['nodes'] = [n for n in data['nodes'] if n['id'] not in nodes_to_remove]
data['components'] = [c for c in data['components'] if c['id'] not in components_to_remove]
data['connections'] = [c for c in data['connections'] if c not in connections_to_remove]


# Add the new T-junction structures
for prefix, start_node_id in paths_to_modify.items():
    current_node_id = start_node_id
    for i in range(1, 4):
        # Create nodes
        tee_node_id = f"{prefix}_tee_{i}"
        outlet_id = f"{prefix}_out_{i}"
        data["nodes"].extend([
            {"id": tee_node_id, "name": tee_node_id, "type": "junction", "elevation": 0.0, "x": 0.0, "y": 0.0},
            {"id": outlet_id, "name": outlet_id, "type": "outlet", "elevation": 0.0, "x": 0.0, "y": 0.0}
        ])

        # Create components
        pipe_id = f"{prefix}_pipe_{i}"
        nozzle_id = f"{prefix}_nozzle_{i}"
        data["components"].extend([
            {"id": pipe_id, "name": pipe_id, "type": "channel", "length": 0.1, "diameter": 0.02},
            {"id": nozzle_id, "name": nozzle_id, "type": "nozzle", "diameter": 0.0035, "nozzle_type": "rounded"}
        ])

        # Create connections
        data["connections"].extend([
            {"from_node": current_node_id, "to_node": tee_node_id, "component": pipe_id},
            {"from_node": tee_node_id, "to_node": outlet_id, "component": nozzle_id}
        ])
        current_node_id = tee_node_id

    # Final nozzle
    final_outlet_id = f"{prefix}_final_out"
    final_nozzle_id = f"{prefix}_final_nozzle"
    data["nodes"].append(
        {"id": final_outlet_id, "name": final_outlet_id, "type": "outlet", "elevation": 0.0, "x": 0.0, "y": 0.0}
    )
    data["components"].append(
        {"id": final_nozzle_id, "name": final_nozzle_id, "type": "nozzle", "diameter": 0.0035, "nozzle_type": "rounded"}
    )
    data["connections"].append(
        {"from_node": current_node_id, "to_node": final_outlet_id, "component": final_nozzle_id}
    )

# Save the modified network
with open('C:/code/lubrication3/complex_network_modified.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Successfully created complex_network_modified.json")
