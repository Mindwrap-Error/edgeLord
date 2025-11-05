#include <bits/stdc++.h>
#include "json.hpp"
using namespace std;

using json = nlohmann::json;

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
    
    static void from_json(json& j, Graph& g);

    json remove_edge(const json& q1);

    json mod_edge(const json& q2);

    json shortest_path(const json& q3);

    json knn(const json& q4);
};