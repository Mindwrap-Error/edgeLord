#include "Graph.hpp"
using json = nlohmann::json;
using std::string;
using std::vector;
using std::pair;
using std::priority_queue;
using std::greater;
using std::map;
using std::unordered_set;

int type_to_int(string& s){
    int type=-1;
    if(s == "local") type = 0;
    else if(s == "primary") type = 1;
    else if(s == "secondary") type = 2;
    else if(s == "tertiary") type = 3;
    else if(s == "expressway") type = 4;
    return type;
}

int poi_to_int(string& s){
    int poi;
    if(s == "restaurant") poi = 0;
    else if(s == "petrol station") poi = 1;
    else if(s == "hospital") poi = 2;
    else if(s == "pharmacy") poi = 3;
    else if(s == "hotel") poi = 4;
    else if(s == "atm") poi = 5;
    return poi;
}

//function in node class
void Node::from_json(json& j,Node& n){
    n.node_id = j.at("id").get<int>();
    n.lat = j.at("lat").get<double>();
    n.lon = j.at("lon").get<double>();
    n.pois = vector<bool>(6,false);

    if (j.contains("pois")) {
        for(const auto& x : j.at("pois")){
            string s = x.get<string>();
            n.pois[poi_to_int(s)] = true;
        }
    }
    //to avoid errors since pois might not be there
}

//function in edge class
void Edge::from_json(json& j,Edge& e){
    e.edge_id = j.at("id").get<int>();
    e.node1 = j.at("u").get<int>();
    e.node2 = j.at("v").get<int>();
    e.len = j.at("length").get<double>();
    e.avg_time = j.at("average_time").get<double>();
    e.oneway = j.at("oneway").get<bool>();
    e.disabled = false; //enable by default
    string s = j.at("road_type").get<string>();
    e.type = type_to_int(s);

    if (j.contains("speed_profile")) {
        for(const auto& x : j.at("speed_profile")){
            e.spd_profile.push_back(x.get<double>());
        }
    }
    //again only do when it contains spd profile
}

//functions in graph class
void Graph::from_json(const json& j,Graph& g){
    g.num_nodes = j["meta"]["nodes"].get<int>();
    int o = g.num_nodes;
    
    for(auto x : j.at("nodes")){
        Node l;
        Node::from_json(x,l);
        g.nodes.push_back(l);
    }
    
    //should we resize and use [] instead of push_back
    int tmp = j.at("edges").size();
    g.adjlist.resize(tmp);

    for(auto y : j.at("edges")){
        Edge e;
        Edge::from_json(y,e);
        g.edges[e.edge_id] = e;
        g.adjlist[e.node1].push_back({e.node2,e.edge_id});
        if(!e.oneway) g.adjlist[e.node2].push_back({e.node1,e.edge_id});
    }
    
    // std::cerr << "graph initialized" << std::endl;
    return;
}



pair<vector<double>, vector<int>> Graph::dijkstra(int source, const string& mode, const vector<bool>& forbidden_nodes, const vector<bool>& not_forbidden_types) {
    vector<double> dist(num_nodes,DBL_MAX);
    vector<int> parent(num_nodes,-1);
    
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq; //greater used cos min queue
    //minimum priority-queue with pair as cost and node_id
    dist[source] = 0;
    pq.push({0.00, source});

    while(!pq.empty()){
        double cost = pq.top().first;
        int u = pq.top().second;
        pq.pop();

        if (cost > dist[u]) continue; //older value

        if (forbidden_nodes[u]) continue;
        for(auto& node : adjlist[u]){
            int v = node.first;
            int edge_id = node.second;

            Edge& edge = edges.at(edge_id);
            if (edge.disabled) continue;
            if (!not_forbidden_types[edge.type]) continue;
            if (forbidden_nodes[v]) continue;

            double edge_cost;
            edge_cost = edge.len;
            if (edge_cost == DBL_MAX) continue;

            //dijkstra logic
            if (dist[u] + edge_cost < dist[v]){
                dist[v] = dist[u] + edge_cost;
                parent[v] = u;
                pq.push({dist[v],v});
            }
        }
    }
    return {dist, parent};
}

double Graph::heuristic(int u, int target)
{
    // use straight-line (Euclidean) distance on lat/lon as a simple heuristic
    double lat1 = nodes[u].lat;
    double lon1 = nodes[u].lon;
    double lat2 = nodes[target].lat;
    double lon2 = nodes[target].lon;

    // approximate conversion of degree to metres: 1 degree lat = 111,000 metres
    double d_lat = abs(lat1 - lat2) * 111000;
    double d_lon = abs(lon1 - lon2) * 111000 * cos(lat1 * M_PI / 180.0);

    return sqrt(d_lat * d_lat + d_lon * d_lon);
}

double Graph::weighted_Astar(int source, int target, double w)
{
    
    if(source == target)
    {
        return 0.0;
    }

    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;

    vector<double> g(num_nodes, DBL_MAX);
    vector<bool> closed(num_nodes, false);
    g[source] = 0.0;

    double h_initial = heuristic(source, target);
    pq.push({h_initial * w, source});

    while(!pq.empty())
    {
        int u = pq.top().second;
        pq.pop();

        if(closed[u])
        {
            continue;
        }
        closed[u] = true;

        if(u==target)
        {
            return g[target];
        }


        for(auto& linked : adjlist[u])
        {
            int v = linked.first;
            int edge_id = linked.second;
            Edge& edge = edges.at(edge_id);
            if(edge.disabled) continue;

            double weight = edge.len;
            double guess_g = g[u] + weight;

            if(guess_g < g[v])
            {
                g[v] =  guess_g;
                double new_f = guess_g + w * heuristic(v, target);
                pq.push({new_f, v});
            }
        }
    }

    return DBL_MAX; //path not found
}

vector<int> Graph::reconstruct_path(int source, int target, const vector<int>& parent)
{
    vector<int> path;
    int current_node = target;
    while(current_node != -1)
    {
        path.push_back(current_node);
        if (current_node == source) break;
        current_node = parent[current_node];
    }

    if(path.empty() || path.back() != source)
    {
        return {};
    }

    reverse(path.begin(), path.end());
    return path;
}

int Graph::get_edge_id(int u, int v)
{
    if (u >= adjlist.size()) return -1;

    for (const auto& pair : adjlist[u])
    {
        if (pair.first == v)
        {
            return pair.second;
        }
    }

    return -1;
}

double Graph::calculate_path_distance(const vector<int>& path)
{
    double total_distance = 0.0;
    for (size_t i = 0; i < path.size() - 1; i++)
    {
        int u = path[i];
        int v = path[i+1];

        int edge_id = get_edge_id(u, v);

        if(edge_id == -1 || edges.find(edge_id) == edges.end())
        {
            return DBL_MAX;
        }

        total_distance += edges.at(edge_id).len;
    }

    return total_distance;
}

json Graph::k_shortest_paths_exact(const json& query,json& response)
{
    int source = query.at("source").get<int>();
    int target = query.at("target").get<int>();
    int k = query.at("k").get<int>();
    string mode = "distance";

    response["id"] = query.at("id");
    json& paths_json = response["paths"] = json::array();

    vector<vector<int>> paths;

    using PathCandidate = pair<double, vector<int>>;
    priority_queue<PathCandidate, vector<PathCandidate>, greater<PathCandidate>> candidates_pq;

    vector<bool> initial_forbidden_nodes(num_nodes, false);
    vector<bool> initial_not_forbidden_types(5, true);
    auto [dist, parent] = dijkstra(source, mode, initial_forbidden_nodes, initial_not_forbidden_types);

    if(dist[target] == DBL_MAX)
    {
        return response;
    }

    vector<int> first_path = reconstruct_path(source, target, parent);
    paths.push_back(first_path);

    paths_json.push_back({
        {"path", first_path},
        {"length", dist[target]}
    });

    for (size_t i = 1; i < k; i++) {
        const vector<int>& previous_path = paths[i - 1];

        for (size_t j = 0; j < previous_path.size() - 1; j++) {
            int spur_node = previous_path[j];
            vector<int> root_path(previous_path.begin(), previous_path.begin() + j + 1);

            vector<int> edges_to_disable;
            vector<bool> forbidden_nodes_dijkstra(num_nodes, false);

            // prune edges that are part of previous shortest paths
            for (const auto& p : paths) {
                if (p.size() > j + 1) {
                    vector<int> p_root(p.begin(), p.begin() + j + 1);
                    if (p_root == root_path) {
                        int edge_id = get_edge_id(p[j], p[j + 1]);
                        if (edge_id != -1 && !edges.at(edge_id).disabled) {
                            edges_to_disable.push_back(edge_id);
                            edges.at(edge_id).disabled = true;
                        }
                    }
                }
            }

            for (size_t node_idx = 0; node_idx < j; ++node_idx) {
                forbidden_nodes_dijkstra[previous_path[node_idx]] = true;
            }

            auto [spur_dist, spur_parent] = dijkstra(spur_node, mode, forbidden_nodes_dijkstra, initial_not_forbidden_types);

            if (spur_dist[target] != DBL_MAX) {
                vector<int> spur_path = reconstruct_path(spur_node, target, spur_parent);
                
                vector<int> total_path = root_path;
                total_path.insert(total_path.end(), spur_path.begin() + 1, spur_path.end());

                double total_distance = calculate_path_distance(total_path);

                if (total_distance != DBL_MAX) {
                    candidates_pq.push({total_distance, total_path});
                }
            }

            for (int edge_id : edges_to_disable) {
                edges.at(edge_id).disabled = false;
            }
        } 

        if (candidates_pq.empty()) {
            break;
        }

        PathCandidate best_candidate;
        bool found_new_path = false;

        while (!candidates_pq.empty()) {
            best_candidate = candidates_pq.top();
            candidates_pq.pop();
            
            bool already_found = false;
            for (const auto& p : paths) {
                if (p == best_candidate.second) {
                    already_found = true;
                    break;
                }
            }
            
            if (!already_found) {
                found_new_path = true;
                break;
            }
        }

        if (!found_new_path) {
            break;
        }

        paths.push_back(best_candidate.second);
        paths_json.push_back({
            {"path", best_candidate.second},
            {"length", best_candidate.first}
        });
    }

    return response;
}

json Graph::k_shortest_paths_heuristic(const json& query,json& response)
{
    //get data from query
    int source = query.at("source").get<int>();
    int target = query.at("target").get<int>();
    int k = query.at("k").get<int>();
    string mode = "distance"; //we only do distance for now


    response["id"] = query.at("id");
    response["paths"] = json::array();

    //these are out parameters for the heuristic
    const double PENALTY_FACTOR = 1.2; 
    const double MAX_STRETCH = 1.5; 
    const double OVERLAP_THRESHOLD = query.at("overlap_threshold").get<double>();

    auto reconstruct_path = [&](int s, int t, const vector<int>& p) -> vector<int> {
        if (p[t] == -1 && s != t) return {}; //no path found
        vector<int> ans;
        for(int v=t;v!=-1;v=p[v]) {
            ans.push_back(v);
            if(v == s) break; 
        }
        reverse(ans.begin(),ans.end());
        return ans;
    };

    auto calculate_real_distance = [&](const vector<int>& path, const map<int, double>& modifs) -> double {
        double dist = 0;
        for (size_t i=0;i<path.size()-1;i++) {
            int eid = get_edge_id(path[i], path[i+1]);
            if (eid != -1) {
                if (modifs.count(eid)) dist += modifs.at(eid); //use original length if modified
                else dist += edges.at(eid).len;
            }
        }
        return dist;
    };

    struct Candidate {
        vector<int> path;
        double length;
        unordered_set<int> edges; //needed to check overlap quickly
    };
    vector<Candidate> candidates; //store candidate paths to choose from

    map<int, double> orig_len; //map from edgeid to original length

    vector<bool> forbidden_nodes(num_nodes,false);
    vector<bool> not_forbidden_types(5, true); //all allowed

    auto [dist, parents] = dijkstra(source,mode,forbidden_nodes,not_forbidden_types);

    if(dist[target] == DBL_MAX) return response; 

    //first path is always optimal so we add it
    vector<int> first_path = reconstruct_path(source, target, parents);
    double optimal_dist = dist[target];

    unordered_set<int> first_edges;
    for(size_t i=0; i<first_path.size()-1; ++i) {
        int eid = get_edge_id(first_path[i], first_path[i+1]);
        if(eid!=-1) first_edges.insert(eid);
    }
    candidates.push_back({first_path, optimal_dist, first_edges});


    //add weights penalty to edges in first path
    for(int eid : first_edges) {
        if(orig_len.find(eid) == orig_len.end()) orig_len[eid] = edges[eid].len;
        edges[eid].len *= PENALTY_FACTOR;
    }

    int attempts = 0;
    //retry are allowed if we get invalid or duplicate paths

    while(candidates.size()<k*2 && attempts<k*5) {
        attempts++;
        auto [dists, parents] = dijkstra(source, mode, forbidden_nodes, not_forbidden_types);
        if (dists[target] == DBL_MAX) break;

        vector<int> new_path = reconstruct_path(source, target, parents);
        double real_dist = calculate_real_distance(new_path, orig_len);

        bool unique = true;
        for(const auto& c : candidates) {
            if(c.path == new_path) { 
                unique = false; 
                break; 
            }
        }

        unordered_set<int> new_edges;
        for(size_t i=0; i<new_path.size()-1; i++) {
            int eid = get_edge_id(new_path[i], new_path[i+1]);
            if(eid != -1) new_edges.insert(eid);
        }
        //we penalize duplicates and retry so no infinite loops
        if(!unique) {
            for(int eid : new_edges) {
                if(orig_len.find(eid) == orig_len.end()) orig_len[eid] = edges[eid].len;
                edges[eid].len *= PENALTY_FACTOR;
            }
            continue;
        }
        bool accept = false;
        if(candidates.size() < k) accept = true; 
        else if(real_dist <= (optimal_dist*MAX_STRETCH)) accept = true;
        //so we  guarantee k candidates within stretch limit
        if(accept) {
            unordered_set<int> new_edges;
            for(size_t i=0;i<new_path.size()-1;i++) {
                int eid = get_edge_id(new_path[i], new_path[i+1]);
                if(eid!=-1) new_edges.insert(eid);
            }
            candidates.push_back({new_path, real_dist, new_edges});
            
            //penalty to found path
            for (int eid : new_edges) {
                if (orig_len.find(eid) == orig_len.end()) orig_len[eid] = edges[eid].len;
                edges[eid].len *= PENALTY_FACTOR;
            }
        }
    }

    //restore edges to their original lengths
    for (auto const& [eid, original_len] : orig_len) {
        edges.at(eid).len = original_len;
    }

    //we will now select the best k path based on penalty scores
    vector<int> selected = {0}; //shortest path always

    while(selected.size()<k && selected.size()<candidates.size()) {
        int best_ind = -1;
        double min_score = DBL_MAX;

        for(size_t i=1;i<candidates.size();i++) {
            bool present = false;
            for(int s : selected) if(s == (int)i) present = true;
            if(present) continue;

            //what if we added i
            vector<int> test_group = selected;
            test_group.push_back(i);
            
            double total_pen = 0;
            if(best_ind == -1) best_ind = i;
            // Sum penalties for every path in this hypothetical group
            for(int ind : test_group) {
                const auto& curr = candidates[ind];
                
                //distance penalty
                double dist_pen = ((curr.length-optimal_dist)/optimal_dist) + 0.1;

                //we calculate overall penlty as given in statement
                int overlap_pen = 0;
                for(int ind1 : test_group) {
                    int common = 0;
                    for(int eid : curr.edges) { //here we include itself as given
                        if(candidates[ind1].edges.count(eid)) common++; //if present in both
                    }
                    
                    double total_overlap = 0;
                    if(!curr.edges.empty()) total_overlap = (double)common/curr.edges.size();
                    if(total_overlap > OVERLAP_THRESHOLD) overlap_pen++;
                }

                total_pen += (dist_pen*overlap_pen);
            }

            if(total_pen<min_score) { //here we minimize total penalty as required
                min_score = total_pen;
                best_ind = i;
            }
        }

        if(best_ind != -1) selected.push_back(best_ind);
        else break;
    }

    //get all final candidates
    vector<Candidate> valid_cand;
    for(int idx : selected) valid_cand.push_back(candidates[idx]);

    //sort by using custom comparator (which is order of length)
    sort(valid_cand.begin(), valid_cand.end(), [](const Candidate& a, const Candidate& b){
        return a.length < b.length;
    });

    for(const auto& c : valid_cand) {
        response["paths"].push_back({
            {"path", c.path},
            {"length", c.length}
        });
    }

    return response;
}
json Graph::approx_shortest_path(const json& query,json& response) 
{
    // placeholder implementation to ensure compilation; implement approximation later
    int id = query.at("id").get<int>();
    double time_budget = query.at("time_budget_ms").get<double>();
    double w = 2;
    response["id"] = id;
    vector<json> distances;

    auto start_time = std::chrono::high_resolution_clock::now();

    for(const auto& q : query.at("queries"))
    {
        int source = q.at("source").get<int>();
        int target = q.at("target").get<int>();

        auto curr_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(curr_time - start_time).count();

        if(elapsed >= time_budget)
        {
            break;
        }

        double dist = weighted_Astar(source, target, w);

        if(dist != DBL_MAX)
        {
            distances.push_back({
                {"source", source},
                {"target", target},
                {"approx_shortest_distance", dist}
            });
        }
    }
    response["distances"] = distances;
    return response;
}


json Graph::process_query(const json& query,json& answer){
    string s = query.at("type").get<string>();
    if(s == "k_shortest_paths") return k_shortest_paths_exact(query,answer);
    if(s == "k_shortest_paths_heuristic") return k_shortest_paths_heuristic(query,answer);
    if(s == "approx_shortest_path") return approx_shortest_path(query,answer);
}