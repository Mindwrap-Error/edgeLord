#include<bits/stdc++.h>
using namespace std;

struct Node{
    int node_id;
    double lat;
    double lon;
    vector<string> pois;
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
};

class Graph{
    int num_nodes;
    vector<Node> nodes;
    map<int,Edge> edges;// edge_id -> edge
    vector<vector<int>> adjlist;
    //adjlist[node1_id][node2_id] = edge_id

    Graph(){
        
    };

};