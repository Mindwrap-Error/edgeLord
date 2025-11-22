#include "Graph.hpp"
using json = nlohmann::json;
using std::string;
using std::vector;
using std::pair;
using std::priority_queue;
using std::greater;
using std::map;


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
    g.graphpois.resize(6);
    
    for(auto x : j.at("nodes")){
        Node l;
        Node::from_json(x,l);
        g.nodes.push_back(l);
    }
    
    g.adjlist.resize(o);

    for(auto y : j.at("edges")){
        Edge e;
        Edge::from_json(y,e);
        g.edges[e.edge_id] = e;
        g.adjlist[e.node1].push_back({e.node2,e.edge_id});
        if(!e.oneway) g.adjlist[e.node2].push_back({e.node1,e.edge_id});
    }
    
    for(int i = 0; i < o;i++){
        for(int j = 0;j < 6;j++){
            if(g.nodes[i].pois[j]) g.graphpois[j].push_back(i);
        }
    }

    // cerr << "graph initialized" << endl;
    return;
}

//ret false if edge dne or edge alr disabled
json Graph::remove_edge(const json& q1,json& answer){
    int id_rem = q1.at("edge_id").get<int>();
    answer["id"] = q1.at("id").get<int>();

    if (edges.find(id_rem) == edges.end()) { //is there edge which we need to remove
        answer["done"] = false;
        return answer;
    }
    if (edges[id_rem].disabled) {
        answer["done"] = false;
    } else {
        edges[id_rem].disabled = true;
        answer["done"] = true;
    }
    
    return answer;
};

json Graph::mod_edge(const json& q2,json& answer){
    int mod_id = q2.at("edge_id");// id of edge to be modified

    // the following variables tell what has been modified and what has not been modified
    bool mod_len = true; // true if length has been modified and false if not
    bool mod_avg_time = true;
    bool mod_spd_profile = true;
    bool mod_type = true;

    //only come into play if the corresponding mod varialble is true;
    double new_len;
    double new_avg_time;
    vector<double> new_spd_profile;
    int new_type;

    if(q2.contains("patch")){
    if (q2["patch"].contains("length")) {
        new_len = q2["patch"]["length"].get<double>();
    }else {
        mod_len = false;
    }
    if(q2["patch"].contains("average_time")){
        new_avg_time = q2["patch"].at("average_time").get<double>();
    } else{
        mod_avg_time = false;
    }
    if(q2["patch"].contains("road_type")){
        string s = q2["patch"].at("road_type").get<string>();
        new_type = type_to_int(s);
    } else{
        mod_type = false;
    }
    if(q2["patch"].contains("speed_profile")){
        auto k = q2["patch"].at("speed_profile");
        for(auto l : k){
            new_spd_profile.push_back(l.get<double>());
        }
    } else{
        mod_spd_profile = false;
    }
}else{
    mod_len = mod_avg_time = mod_type = mod_spd_profile = false;
}
        
    answer["id"] = q2.at("id").get<int>();

    if (edges.find(mod_id) == edges.end()) {
        answer["done"] = false; //edge not found
        return answer;
    }

    Edge& mod_edge = edges.find(mod_id)->second;
    bool changed = mod_len || mod_avg_time || mod_spd_profile || mod_type;

    
    if (mod_edge.disabled) { //first enable the edge if disabled
        mod_edge.disabled = false; 
        answer["done"] = true;
    } else {
        if (!changed) {
            answer["done"] = false; //ret false if we not changing anything at all
        } else {
            answer["done"] = true;
        }
    }
    if (changed && !answer["done"]) { // modify respectively
        if (mod_len) mod_edge.len = new_len;
        if (mod_avg_time) mod_edge.avg_time = new_avg_time;
        if (mod_type) mod_edge.type = new_type;
        if (mod_spd_profile) mod_edge.spd_profile = new_spd_profile;
    }
    return answer;
};


double Graph::calc_timecost(const Edge& edge, double T_arrival) {
    // T_arrival is in seconds and could be more than a day
    int days = T_arrival / 86400.0;
    T_arrival = T_arrival - 86400.0*days;
    //if no speed profile
    if (edge.spd_profile.empty()) {
        return edge.avg_time;
    }

    double remains = edge.len;
    double ans = 0.0;


    int slot = ((int) T_arrival/900 )% 24;
    double dist_in_first_slot = edge.spd_profile[slot] * ((1+slot)*900.0-T_arrival);
    if(remains <= dist_in_first_slot){
        return remains/edge.spd_profile[slot];
    }
    else{
        remains -= dist_in_first_slot;
        ans += 900.0;
        slot = (slot + 1)%24;
    }

    while(true){
        double dist_in_this_slot = edge.spd_profile[slot] * 900.0;
        if(remains > dist_in_this_slot){
            remains -= dist_in_this_slot;
            ans+=900.0;
            slot = (slot+1)%24;
            continue;
        } else{
            ans += remains / edge.spd_profile[slot];
            remains = 0;
            break;
        }
    }
    return ans;
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
            if (mode == "distance") edge_cost = edge.len;
            else edge_cost = calc_timecost(edge,dist[u]);
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


json Graph::shortest_path(const json& q3,json& answer){

    int source = q3.at("source").get<int>();
    int target = q3.at("target").get<int>();

    string mode = q3.at("mode").get<string>(); // "time" or "distance"

    vector<bool> forbidden_nodes(num_nodes, false);
    vector<bool> not_forbidden_types(5, true);// true == not forbidden and false == forbidden

    bool no_constraints = false;// true if no constraints


    if(!q3.contains("constraints")){
        no_constraints = true;
    }

    //similar to no_constrains variable
    bool no_nodes_forbidden = false;
    bool no_types_forbidden = false;

    
    if(!no_constraints){
        if(q3["constraints"].contains("forbidden_road_types")){
            auto x  = q3["constraints"].at("forbidden_road_types");
            for(auto k : x){
                string s = k.get<string>();
                not_forbidden_types[type_to_int(s)] = false;
            }
        } else{
            no_types_forbidden = true;
        }
        if(q3["constraints"].contains("forbidden_nodes")){
            auto y = q3["constraints"].at("forbidden_nodes");
            for(auto l : y){
                forbidden_nodes[l.get<int>()] = true; 
            }
        } else{
            no_nodes_forbidden = true;
        }
    }
    if(no_nodes_forbidden && no_types_forbidden) no_constraints = true;
    //final path in this vector
    vector<int> final_path;

    // this varible set to false if no path exists
    bool possible = true;

    //mintime/dist -> this set to minimum time or distance as asked by mode
    double mincost;

    auto [dist, parent] = dijkstra(source, mode, forbidden_nodes, not_forbidden_types);

    if (dist[target] == DBL_MAX) {
        possible = false;
    } else {
        possible = true;
        mincost = dist[target];

        int tmp = target;
        while(tmp != -1) {
            final_path.push_back(tmp);
            tmp = parent[tmp];
        }
        reverse(final_path.begin(),final_path.end());
    }

    answer["id"] = q3.at("id").get<int>();
    answer["possible"] = possible;

    if(possible){
        answer["path"] = final_path;
        answer["minimum_time/minimum_distance"] = mincost;
    }
    return answer;

}

json Graph::knn(const json& q4,json& answer){

    string s = q4.at("poi").get<string>();
    int req_poi = poi_to_int(s);

    //query point latitude and longitude
    double qp_lat = q4.at("query_point").at("lat").get<double>();
    double qp_lon = q4.at("query_point").at("lon").get<double>();
    
    //k
    int k = q4.at("k").get<int>();

    int metric;// = 0->euclidian 1-> shortest path
    s = q4.at("metric").get<string>();
    if(s == "euclidean") metric = 0;
    else if(s == "shortest_path") metric = 1;

    //store final nodes in this vector
    vector<int> final_nodes;

    priority_queue<pair<double, int>> max_heap; //for k-nearest

    
    vector<int> poi_nodes = graphpois[req_poi]; //stores all nodes which have the poi needed

    if (metric==0) {//euclidean
        for (int node : poi_nodes) {
            double d = (qp_lat-nodes[node].lat) * (qp_lat-nodes[node].lat) + (qp_lon-nodes[node].lon) * (qp_lon-nodes[node].lon);
            max_heap.push({d,node});
            if (max_heap.size() > k) {
                max_heap.pop();
            }
        }
    } else {//shortest path
        int nearest_node = -1;
        double min_dist_sq = DBL_MAX;
        for (int i = 0; i < num_nodes; ++i) {
            double d = (qp_lat-nodes[i].lat) * (qp_lat-nodes[i].lat) + (qp_lon-nodes[i].lon) * (qp_lon-nodes[i].lon);
            if (d < min_dist_sq) {
                min_dist_sq = d;
                nearest_node = i;
            }
        }

        //run dijkstra from this node now
        if (nearest_node != -1) {
            vector<bool> forbidden_nodes(num_nodes, false);
            vector<bool> not_forbidden_types(5, true);
            //we use distance as metric in this 
            auto [dist, parent] = dijkstra(nearest_node, "distance", forbidden_nodes, not_forbidden_types);

            //best k nodes are selected
            for (int poi_node : poi_nodes) {
                double d = dist[poi_node];
                if (d != DBL_MAX) { //if reachable
                    max_heap.push({d, poi_node});
                    if (max_heap.size() > k) max_heap.pop();
                }
            }
        }
    }
    //extract and store in final_nodes
    while(!max_heap.empty()) {
        final_nodes.push_back(max_heap.top().second);
        max_heap.pop();
    }

    //update answer
    answer["id"] = q4.at("id").get<int>();
    reverse(final_nodes.begin(), final_nodes.end());
    answer["nodes"] = final_nodes;

    return answer;
}


json Graph::process_query(const json& query,json& answer){
    // cerr << x++ << endl;
    string s = query.at("type").get<string>();
    if (s == "knn") return knn(query,answer);
    else if(s == "shortest_path") return shortest_path(query,answer);
    else if(s == "modify_edge") return mod_edge(query,answer);
    else if(s == "remove_edge") return remove_edge(query,answer);
}