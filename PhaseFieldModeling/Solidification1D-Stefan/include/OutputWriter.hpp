#pragma once

#include <string>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "Mesh.hpp"

// Ensure these headers are in your include/ folder
#include "json.hpp" 
#include "httplib.h"

using json = nlohmann::json;

template <unsigned int Nsd, unsigned int Nne>
class OutputWriter {
public:
    OutputWriter(const Mesh<Nsd, Nne>& mesh, const std::string& outDir, const std::string& serverHost, int serverPort)
        : mesh_(mesh), outputDirectory_(outDir), host_(serverHost), port_(serverPort) {
        std::filesystem::create_directories(outputDirectory_);
    }

    void writeAndSend(int timestep, double time, const Eigen::VectorXd& phi, const Eigen::VectorXd& T) const {
        unsigned int num_nodes = mesh_.Nnodes();
        
        // 1. Prepare data containers
        std::vector<double> x_coords(num_nodes);
        std::vector<double> phi_vals(num_nodes);
        std::vector<double> T_vals(num_nodes);

        // 2. Extract data from mesh and vectors
        for (unsigned int i = 0; i < num_nodes; ++i) {
            x_coords[i] = mesh_.nodes[i].x1;
            phi_vals[i] = phi(i);
            T_vals[i] = T(i);
        }

        // 3. Serialize to JSON
        json payload;
        payload["timestep"] = timestep;
        payload["time"] = time;
        payload["x"] = x_coords;
        payload["phi"] = phi_vals;
        payload["T"] = T_vals;

        // 4. Send to FastAPI Server via HTTP POST
        httplib::Client cli(host_, port_);
        std::string json_str = payload.dump();
        
        auto res = cli.Post("/update", json_str, "application/json");
        
        // Basic error checking
        if (!res) {
            std::cerr << "Warning: Could not connect to plotting server at " << host_ << ":" << port_ << std::endl;
        } else if (res->status != 200) {
            std::cerr << "Warning: Server returned status " << res->status << std::endl;
        }
    }

private:
    const Mesh<Nsd, Nne>& mesh_;
    std::string outputDirectory_;
    std::string host_;
    int port_;
};