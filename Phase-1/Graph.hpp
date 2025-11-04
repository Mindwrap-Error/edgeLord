#include <bits/stdc++.h>
#include "json.hpp"
using namespace std;

using json = nlohmann::json;

struct Node{
    int node_id;
    double lat;
    double lon;
    vector<string> pois;

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
    int num_nodes;
    vector<Node> nodes;
    map<int,Edge> edges;// edge_id -> edge
    vector<vector<pair<int,int>>> adjlist;
    //adjlist[node1_id] = {{node2,edge_id},...}

    static void from_json(json& j, Graph& g);

};