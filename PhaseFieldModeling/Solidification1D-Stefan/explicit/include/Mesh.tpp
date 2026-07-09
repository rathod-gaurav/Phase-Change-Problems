#pragma once

#include "Mesh.hpp"
#include <fstream>
#include <string>
#include <filesystem>

template <unsigned int Nsd, unsigned int Nne>
void Mesh<Nsd,Nne>::writeToFiles(const std::string& dir) const {
    std::filesystem::create_directories(dir); //creates dir folder if it does not exist

    std::ofstream points_file(dir + "/points.txt");
    for(const auto& node : nodes){
        if constexpr (Nsd == 1){
            points_file << node.x1 << "\n";
        }
    }

    std::ofstream elems_file(dir + "/elems.txt"); //element triangulation file
    for(const auto& elem : elements){
        for(unsigned int i = 0 ; i < Nne; ++i){
            elems_file << elem.node[i] << " ";
        }
        elems_file << "\n";
    }
}