import networkx as nx
import random
from infomap import Infomap

# Parameters
num_layers = 3
nodes = list(range(1, 11))
inter_layer_weight = 1.0



# Create test data before reading
u_max=1000
l_max=3

# Option to add networkx graph using add_networkx_graph()

# Initialize Infomap in multilayer mode
im = Infomap("--two-level")

# Add coupling edges
for u in range(1,u_max+1):
	for l1 in range(1,l_max+1):
		for l2 in range(l1+1,l_max+1):
			# Link format: source_l,node_id,target_l
			im.add_multilayer_inter_link(l1,u,l2)

# Generate random intra-layer links. fifty per layer.
for l in range(1,l_max+1):
	for _ in range(2000):
		u1=random.randint(1,u_max)
		u2=random.randint(1,u_max)

		# Add link. Format: (src_l,src_u),(trg_l,trg_u)
		im.add_multilayer_link((l,u1),(l,u2))



# Run Infomap
print("Running Infomap on multilayer network...")
im.run()

# Collect communities
communities={}
comm_layers={}

# print("\nState node assignments (node, layer) -> module:")
for node in im.nodes:
	physical_id = node.node_id
	layer_id = node.layer_id
	module_id = node.module_id
	# print(f"({physical_id}, layer {layer_id}) -> module {module_id}")

	if physical_id not in communities:
		communities[physical_id] = set()
	communities[physical_id].add(module_id)

	if (physical_id,layer_id) not in comm_layers:
		comm_layers[(physical_id,layer_id)]=set()
	comm_layers[(physical_id,layer_id)].add(module_id)


# Show overlapping nodes
print("\nActors with overlapping community memberships:")
for node, modules in communities.items():
	if len(modules) > 1:
		print(f"Node {node}: Modules {modules}")


# Show overlapping nodes
print("\nNodes with overlapping community memberships:")
for (node,layer), modules in comm_layers.items():
	if len(modules) > 1:
		print(f"Node {(node,layer)}: Modules {modules}")


