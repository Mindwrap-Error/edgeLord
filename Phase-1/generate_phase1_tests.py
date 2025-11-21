import json
import random
import math
import heapq

# --- Configuration ---
NUM_NODES = 5
NUM_EDGES = 10
NUM_QUERIES = 6
BASE_LAT, BASE_LON = 19.07, 72.87
LAT_LON_SCALE = 5.0 # Controls how spread out the nodes are

# --- Constants from Project Spec ---
POIs = ["restaurant", "hospital", "pharmacy", "hotel", "atm", "petrol station"]
ROAD_TYPES = ["primary", "secondary", "tertiary", "local", "expressway"]
METRICS = ["euclidean", "shortest_path"]
MODES = ["time", "distance"]
TIME_SLOTS = 96 # 96 slots of 15 minutes for 24 hours

class Graph:
    """A class to represent and manage the graph for solving queries."""
    def __init__(self, graph_data):
        self.nodes = {node['id']: node for node in graph_data['nodes']}
        self.adj = {node_id: [] for node_id in self.nodes}
        self.edges = {}
        self.inactive_edges = set()
        
        # Helper for quick lookups
        self.poi_map = {poi: [] for poi in POIs}
        for node_id, node in self.nodes.items():
            for poi in node.get('pois', []):
                self.poi_map[poi].append(node_id)

        for edge in graph_data['edges']:
            self._add_edge_internal(edge)
            
    def _add_edge_internal(self, edge):
        u, v = edge['u'], edge['v']
        self.edges[edge['id']] = edge
        self.adj[u].append(v)
        if not edge.get('oneway', False):
            self.adj[v].append(u)
    
    def find_edge_by_nodes(self, u, v):
        """Finds an edge connecting two nodes, if one exists."""
        for edge_id, edge in self.edges.items():
            if edge_id in self.inactive_edges:
                continue
            if (edge['u'] == u and edge['v'] == v) or \
               (edge['u'] == v and edge['v'] == u and not edge.get('oneway', False)):
                return edge
        return None

    def remove_edge(self, edge_id):
        if edge_id in self.edges:
            self.inactive_edges.add(edge_id)
            return True
        return False

    def modify_edge(self, edge_id, patch):
        if edge_id in self.edges:
            # If edge was deleted, restore it
            if edge_id in self.inactive_edges:
                self.inactive_edges.remove(edge_id)
                # If patch is empty, it uses its old values. Otherwise, update.
                if patch:
                    self.edges[edge_id].update(patch)
            else: # Just modify active edge
                if patch:
                    self.edges[edge_id].update(patch)
            return True
        return False
        
    def get_closest_node(self, lat, lon):
        """Finds the node with the minimum Euclidean distance to a lat/lon point."""
        min_dist = float('inf')
        closest_node_id = -1
        for node_id, node in self.nodes.items():
            dist = math.hypot(node['lat'] - lat, node['lon'] - lon)
            if dist < min_dist:
                min_dist = dist
                closest_node_id = node_id
        return closest_node_id

def solve_shortest_path(graph, query):
    """Solves a shortest path query using a time-aware Dijkstra's algorithm."""
    source = query['source']
    target = query['target']
    mode = query['mode']
    constraints = query.get('constraints', {})
    forbidden_nodes = set(constraints.get('forbidden_nodes', []))
    forbidden_road_types = set(constraints.get('forbidden_road_types', []))

    dist = {node_id: float('inf') for node_id in graph.nodes}
    prev = {node_id: None for node_id in graph.nodes}
    dist[source] = 0
    pq = [(0, source)] # (current_cost, node_id)

    while pq:
        d, u = heapq.heappop(pq)

        if u == target:
            break
        if d > dist[u]:
            continue
        if u in forbidden_nodes:
            continue

        for v in graph.adj[u]:
            if v in forbidden_nodes:
                continue

            edge = graph.find_edge_by_nodes(u, v)
            if not edge or edge['id'] in graph.inactive_edges:
                continue
            if edge['road_type'] in forbidden_road_types:
                continue
            
            weight = 0
            if mode == 'distance':
                weight = edge['length']
            elif mode == 'time':
                if 'speed_profile' in edge:
                    current_time_sec = dist[u]
                    time_in_day_sec = current_time_sec % (24 * 3600)
                    slot_duration_sec = 15 * 60
                    
                    time_left_in_slot = slot_duration_sec - (time_in_day_sec % slot_duration_sec)
                    
                    # Simplified time-dependent calculation
                    slot_index = int(time_in_day_sec // slot_duration_sec) % TIME_SLOTS
                    speed = edge['speed_profile'][slot_index]
                    
                    time_to_travel = edge['length'] / speed if speed > 0 else float('inf')
                    weight = time_to_travel

                else:
                    weight = edge['average_time']

            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    # Reconstruct path
    if dist[target] == float('inf'):
        return {"id": query['id'], "possible": False}
    
    path = []
    curr = target
    while curr is not None:
        path.append(curr)
        curr = prev[curr]
    path.reverse()
    
    result_key = "minimum_distance" if mode == 'distance' else "minimum_time"
    return {
        "id": query['id'],
        "possible": True,
        result_key: dist[target],
        "path": path
    }

def solve_knn(graph, query):
    """Solves a K-Nearest Neighbors query."""
    poi = query['poi']
    k = query['k']
    metric = query['metric']
    
    candidate_nodes = graph.poi_map.get(poi, [])
    if not candidate_nodes:
        return {"id": query['id'], "nodes": []}

    results = []
    if metric == 'euclidean':
        query_point = query['query_point']
        lat, lon = query_point['lat'], query_point['lon']
        
        distances = []
        for node_id in candidate_nodes:
            node = graph.nodes[node_id]
            dist = math.hypot(node['lat'] - lat, node['lon'] - lon)
            distances.append((dist, node_id))
        
        distances.sort()
        results = [node_id for dist, node_id in distances[:k]]

    elif metric == 'shortest_path':
        query_point = query['query_point']
        # Per spec, find nearest node to lat/lon and calculate shortest paths from there
        source_node = graph.get_closest_node(query_point['lat'], query_point['lon'])
        
        # Run Dijkstra from this source node to find distances to all other nodes
        dist = {node_id: float('inf') for node_id in graph.nodes}
        dist[source_node] = 0
        pq = [(0, source_node)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue

            for v in graph.adj[u]:
                edge = graph.find_edge_by_nodes(u, v)
                if not edge or edge['id'] in graph.inactive_edges:
                    continue
                
                weight = edge['length'] # Use distance for shortest path metric
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    heapq.heappush(pq, (dist[v], v))
        
        # Filter for candidates, sort by distance, and take top k
        candidate_distances = []
        for node_id in candidate_nodes:
            if dist[node_id] != float('inf'):
                candidate_distances.append((dist[node_id], node_id))
        
        candidate_distances.sort()
        results = [node_id for d, node_id in candidate_distances[:k]]

    return {"id": query['id'], "nodes": results}

def generate_graph():
    """Generates a random graph structure."""
    nodes = []
    for i in range(NUM_NODES):
        nodes.append({
            "id": i,
            "lat": BASE_LAT + (random.random() - 0.5) * LAT_LON_SCALE,
            "lon": BASE_LON + (random.random() - 0.5) * LAT_LON_SCALE,
            "pois": random.sample(POIs, k=random.randint(0, 2))
        })

    edges = []
    edge_id_counter = 1000
    existing_pairs = set()
    while len(edges) < NUM_EDGES:
        u, v = random.sample(range(NUM_NODES), 2)
        if (u, v) in existing_pairs or (v, u) in existing_pairs:
            continue
        existing_pairs.add((u,v))
        
        length = random.uniform(50, 1000)
        avg_speed = random.uniform(5, 15) # m/s
        
        edges.append({
            "id": edge_id_counter,
            "u": u,
            "v": v,
            "length": length,
            "average_time": length / avg_speed,
            "speed_profile": [max(1, s) for s in [avg_speed + random.uniform(-2, 2) for _ in range(TIME_SLOTS)]],
            "oneway": random.choice([True, False]),
            "road_type": random.choice(ROAD_TYPES)
        })
        edge_id_counter += 1

    return {
        "meta": {
            "id": "generated_test_1",
            "nodes": NUM_NODES,
            "description": "Randomly generated test case"
        },
        "nodes": nodes,
        "edges": edges
    }

def generate_queries_and_solutions(graph_obj):
    """Generates a list of queries and their solutions against the graph."""
    queries = []
    solutions = []
    
    all_edge_ids = list(graph_obj.edges.keys())
    
    for i in range(NUM_QUERIES):
        query_type = random.choices(
            ["shortest_path", "knn", "remove_edge", "modify_edge"],
            weights=[0.5, 0.3, 0.1, 0.1], k=1)[0]
        
        query = {"id": i}

        if query_type == "shortest_path" and NUM_NODES > 1:
            source, target = random.sample(range(NUM_NODES), 2)
            query.update({
                "type": "shortest_path",
                "source": source,
                "target": target,
                "mode": random.choice(MODES)
            })
            if random.random() < 0.2: # Add constraints sometimes
                query["constraints"] = {
                    "forbidden_nodes": random.sample(range(NUM_NODES), k=random.randint(1,3)),
                    "forbidden_road_types": random.sample(ROAD_TYPES, k=random.randint(1,2))
                }
            solution = solve_shortest_path(graph_obj, query)

        elif query_type == "knn" and any(graph_obj.poi_map.values()):
            poi = random.choice([p for p, nodes in graph_obj.poi_map.items() if nodes])
            query.update({
                "type": "knn",
                "poi": poi,
                "query_point": {
                    "lat": BASE_LAT + (random.random() - 0.5) * LAT_LON_SCALE,
                    "lon": BASE_LON + (random.random() - 0.5) * LAT_LON_SCALE
                },
                "k": random.randint(2, 5),
                "metric": random.choice(METRICS)
            })
            solution = solve_knn(graph_obj, query)

        elif query_type == "remove_edge" and all_edge_ids:
            edge_to_remove = random.choice(all_edge_ids)
            query.update({"type": "remove_edge", "edge_id": edge_to_remove})
            graph_obj.remove_edge(edge_to_remove) # Apply update
            solution = {"done": True}

        elif query_type == "modify_edge" and all_edge_ids:
            edge_to_modify = random.choice(all_edge_ids)
            query.update({
                "type": "modify_edge",
                "edge_id": edge_to_modify,
                "patch": {"length": round(random.uniform(50, 1000), 2)}
            })
            graph_obj.modify_edge(edge_to_modify, query["patch"]) # Apply update
            solution = {"done": True}

        else:
            continue # Skip if query can't be generated
            
        queries.append(query)
        solutions.append(solution)
        
    return queries, solutions

def main():
    print("Generating graph...")
    graph_json_data = generate_graph()
    with open("graph.json", "w") as f:
        json.dump(graph_json_data, f, indent=2)
    print("-> graph.json created.")

    # Create a graph object to solve queries on the fly
    live_graph = Graph(graph_json_data)
    
    print("Generating queries and solving for expected output...")
    queries, solutions = generate_queries_and_solutions(live_graph)
    
    queries_file_content = {
        "meta": {"id": "qset_generated_1"},
        "events": queries
    }
    with open("queries.json", "w") as f:
        json.dump(queries_file_content, f, indent=2)
    print("-> queries.json created.")
    
    # The spec notes that driver code adds processing time. We will omit it.
    with open("expected_out.json", "w") as f:
        json.dump(solutions, f, indent=2)
    print("-> expected_out.json created.")
    print("\nDone!")

if __name__ == "__main__":
    main()