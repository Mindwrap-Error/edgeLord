#include <bits/stdc++.h>
#include "json.hpp"
using namespace std;
#include "Graph.hpp"

using json = nlohmann::json;

int type_to_int(string& s){
    int type;
    if(s == "local") type = 0;
    else if(s == "primary") type = 1;
    else if(s == "secondary") type = 2;
    else if(s == "tertiary") type = 3;
    else if(s == "expressway") type = 4;
    return type;
}

int poi_to_int(string& s){
    int poi;
    if(s == "restauraunt") poi=0;
    else if(s == "petrol station") poi = 1;
    else if(s == "hospital") poi = 2;
    else if(s == "pharmacy") poi = 3;
    else if(s == "hotel") poi = 4;
    else if(s == "atm") poi = 5;
}


void Node::from_json(json& j,Node& n){
    n.node_id = j.at("id").get<int>();
    n.lat = j.at("lat").get<double>();
    n.lon = j.at("lon").get<double>();
    n.pois = vector<bool>(6,false);

    for(const auto& x : j.at("pois")){
        string s = x.get<string>();
        n.pois[poi_to_int(s)] = true;
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
    e.type = type_to_int(s);

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


json Graph::remove_edge(const json& q1){
    int remov_id = q1.at("edge_id");// id of edge to be removed

        
        // code to remove edge


    json answer;
    answer["id"] = q1.at("id").get<int>();
    answer["done"] = true;
    return answer;
};

json Graph::mod_edge(const json& q2){
    int mod_id = q2.at("edge_id");// id of edge to be modified

    // the following variables tell what has been modified and what has not been modified
    bool mod_len = true; // true if lenght has been modified and false if not
    bool mod_avg_time = true;
    bool mod_spd_profile = true;
    bool mod_type = true;

    //only come into play if the corresponding mod varialble is true;
    double new_len;
    double new_avg_time;
    vector<double> new_spd_profile;
    int new_type;

    //ignore the below code till the next comment
    try{
        new_len = q2["patch"].at("length").get<double>();
    } catch(std::out_of_range){
        mod_len = false;
    }
    try{
        new_avg_time = q2["patch"].at("average_time").get<double>();
    } catch(std::out_of_range){
        mod_avg_time = false;
    }
    try{
        string s = q2["patch"].at("type").get<string>();
        new_type = type_to_int(s);
    } catch(std::out_of_range){
        mod_type = false;
    }
    try{
        auto k = q2["patch"].at("speed_profile");
        for(auto l : k){
            new_spd_profile.push_back(l.get<double>());
        }
    } catch(std::out_of_range){
        mod_spd_profile = false;
    }
        
    //write code to remove edge here with the given variables


    json answer;
    answer["id"] = q2.at("id").get<int>();
    answer["done"] = true;
    return answer;
};

json Graph::shortest_path(const json& q3){

    int source = q3.at("source").get<int>();
    int target = q3.at("target").get<int>();

    string mode = q3.at("mode").get<string>(); // "time" or "distance"

    vector<int> forbidden_nodes; // iski jagah array of bools banana ho toh wo bhi ban jayega
    // vector<bool> forbidden_nodes; uss case mein ye line uncomment karke use kar lena baaki mein dekh lunga
    vector<bool> not_forbidden_types(5, true);// true == not forbidden and false == forbidden

    bool no_constraints = false;// true if no constraints

    //ignore
    try{
        q3.at("constraints");
    } catch(std::out_of_range){
        no_constraints = true;
    }

    //simial to no_constrains variable
    bool no_nodes_forbidden = false;
    bool no_types_forbidden = false;

    //ignore
    if(!no_constraints){
        try{
            auto x  = q3["constraints"].at("forbidden_road_types");
            for(auto k : x){
                string s = k.get<string>();
                not_forbidden_types[type_to_int(s)];
            }
        } catch(std::out_of_range){
            no_types_forbidden = true;
        }
        try{
            auto y = q3["constraints"].at("forbidden_nodes");
            for(auto l : y){
                forbidden_nodes.push_back(l.get<int>());
            }
        } catch(std::out_of_range){
            no_nodes_forbidden = true;
        }
    }

    //store final path in this vector
    vector<int> final_path;

    // set this varible to false if no path exists
    bool possible = true;

    //mintime/dist -> set this to minimum time or distance as asked by mode
    double mincost;

    //----------------write your code here----------------------------------------------------





    //---------------------------------------------------------------------------------------

    json answer;
    answer["id"] = q3.at("id").get<int>();
    answer["possible"] = possible;

    if(possible){
        answer["path"] = final_path;
        answer["minumum_time/minimum_distance"] = mincost;
    }
    return answer;

}

json Graph::knn(const json& q4){

    string s = q4.at("poi").get<string>();
    int reqd_poi = poi_to_int(s);

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

    //------------------------------write your code here------------------------------------------------


    //--------------------------------------------------------------------
    json answer;
    answer["id"] = q4.at("id").get<int>();
    answer["nodes"] = final_nodes;

    return answer;
}