#include <bits/stdc++.h>
#include "json.hpp"
using namespace std;
#include "Graph.hpp"

using json = nlohmann::json;

void Node::from_json(json& j,Node& n){
    n.node_id = j.at("id").get<int>();
    n.lat = j.at("lat").get<double>();
    n.lon = j.at("lon").get<double>();

    for(const auto& x : j.at("pois")){
        n.pois.push_back(x.get<string>());
    }
}

void Edge::from_json(json& j,Edge& e){
    e.edge_id = j.at("id").get<int>();
    e.node1 = j.at("u").get<int>();
    e.node2 = j.at("v").get<int>();
    e.len = j.at("length").get<double>();
    e.avg_time = j.at("average_time").get<double>();
    e.oneway = j.at("oneway").get<bool>();
    e.disabled = true;
    string s = j.at("road_type").get<string>();
    if(s == "local") e.type = 0;
    else if(s == "primary") e.type = 1;
    else if(s == "secondary") e.type = 2;
    else if(s == "tertiary") e.type = 3;
    else if(s == "expressway") e.type = 4;

    for(const auto& x : j.at("speed_profile")){
        e.spd_profile.push_back(x.get<double>());
    }
}

void Graph::from_json(json& j,Graph& g){
    g.num_nodes = j["meta"]["nodes"].get<int>();
    int o = g.num_nodes;

    for(auto x : j.at("nodes")){
        Node l;
        Node::from_json(x,l);
        g.nodes.push_back(l);
    }

    for(auto y : j.at("edges")){
        Edge e;
        Edge::from_json(y,e);
        g.edges[e.edge_id] = e;
        g.adjlist[e.node1].push_back({e.node2,e.edge_id});
        if(!e.oneway) g.adjlist[e.node2].push_back({e.node1,e.edge_id});
    }
}