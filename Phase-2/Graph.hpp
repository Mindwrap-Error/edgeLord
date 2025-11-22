#include<iostream>
#include <nlohmann/json.hpp>
#include <cfloat>       // For DBL_MAX
#include <cmath>        // For _euclidean_dist_sq
#include <unordered_set>
#include <vector>
#include <queue>
#include<string>
#include<map>
using json = nlohmann::json;
using std::string;
using std::vector;
using std::pair;
using std::priority_queue;
using std::greater;
using std::map;

int type_to_int(string& s);
int poi_to_int(string& s);

struct Node{
    int node_id;
    double lat;
    double lon;
    vector<bool> pois;

    static void from_json(json& j, Node& n);
};



struct Edge{
    int edge_id;
    int node1;
    int node2;
    double len;
    double avg_time;
    vector<double> spd_profile;// will be empty if no spd profile
    int type;// primary = 1, secondary = 2, tertiary = 3, local = 0, expressway = 4
    bool oneway;// true if directed edge
    bool disabled;
    static void from_json(json& j, Edge& e);
};

class Graph{
    public:
    int num_nodes;
    vector<Node> nodes;
    map<int,Edge> edges;// edge_id -> edge
    vector<vector<pair<int,int>>> adjlist;
    //adjlist[node1_id] = {{node2,edge_id},...}
    
    static void from_json(const json& j, Graph& g);

    json k_shortest_paths_exact(const json& query);

    json k_shortest_paths_heuristic(const json& query);

    json approx_shortest_path(const json& query);

    private:
    //dijkstra
    pair<vector<double>, vector<int>> dijkstra(int source, const string& mode, const vector<bool>&forbidden_nodes, const vector<bool>& not_forbidden_types);

    vector<int> reconstruct_path(int source, int target, const vector<int>& parent);

    double calculate_path_distance(const vector<int>& path);

    int get_edge_id(int u, int v);

    double heuristic (int u, int target);

    double weighted_Astar(int source, int target, double w);

    json process_query(const json& query);
};