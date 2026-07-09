#pragma once

#include <string>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "Mesh.hpp"
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
            
            std::vector<double> x_coords(num_nodes);
            std::vector<double> phi_vals(num_nodes);
            std::vector<double> T_vals(num_nodes);

            // 1. Extract data and write to local CSV
            std::string filename = outputDirectory_ + "/state_" + std::to_string(timestep) + ".csv";
            std::ofstream file(filename);
            
            if (file.is_open()) {
                file << "x,phi,T\n";
                for (unsigned int i = 0; i < num_nodes; ++i) {
                    x_coords[i] = mesh_.nodes[i].x1;
                    phi_vals[i] = phi(i);
                    T_vals[i] = T(i);
                    file << x_coords[i] << "," << phi_vals[i] << "," << T_vals[i] << "\n";
                }
                file.close();
            } else {
                std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
            }

            // 2. Serialize and Send to Server
            json payload;
            payload["timestep"] = timestep;
            payload["time"] = time;
            payload["x"] = x_coords;
            payload["phi"] = phi_vals;
            payload["T"] = T_vals;

            httplib::Client cli(host_, port_);
            auto res = cli.Post("/update", payload.dump(), "application/json");
        }

    private:
        const Mesh<Nsd, Nne>& mesh_;
        std::string outputDirectory_;
        std::string host_;
        int port_;
};