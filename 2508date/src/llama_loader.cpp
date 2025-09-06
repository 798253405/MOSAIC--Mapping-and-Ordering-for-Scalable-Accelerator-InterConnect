#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>

class LLaMAMatrixLoader {
public:
    int sequence_length;
    int d_head;
    std::vector<std::vector<float>> query_matrix;
    std::vector<std::vector<float>> key_matrix;
    std::vector<std::vector<float>> value_matrix;
    
    bool loadMetadata(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open metadata file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key;
            iss >> key;
            
            if (key == "SEQUENCE_LENGTH") {
                iss >> sequence_length;
            } else if (key == "D_HEAD") {
                iss >> d_head;
            }
        }
        
        std::cout << "Loaded metadata: seq_len=" << sequence_length 
                  << ", d_head=" << d_head << std::endl;
        return true;
    }
    
    bool loadMatrix(const std::string& filename, std::vector<std::vector<float>>& matrix) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open matrix file: " << filename << std::endl;
            return false;
        }
        
        matrix.clear();
        std::string line;
        
        while (std::getline(file, line)) {
            std::vector<float> row;
            std::istringstream iss(line);
            float value;
            
            while (iss >> value) {
                row.push_back(value);
            }
            
            if (!row.empty()) {
                matrix.push_back(row);
            }
        }
        
        std::cout << "Loaded matrix from " << filename 
                  << ": " << matrix.size() << "x" << (matrix.empty() ? 0 : matrix[0].size()) 
                  << std::endl;
        return true;
    }
    
    bool loadAllMatrices(const std::string& input_dir = "./Input/") {
        // Load metadata first
        if (!loadMetadata(input_dir + "llama_metadata.txt")) {
            return false;
        }
        
        // Load Q, K, V matrices
        if (!loadMatrix(input_dir + "llama_query.txt", query_matrix)) {
            return false;
        }
        if (!loadMatrix(input_dir + "llama_key.txt", key_matrix)) {
            return false;
        }
        if (!loadMatrix(input_dir + "llama_value.txt", value_matrix)) {
            return false;
        }
        
        std::cout << "\nAll matrices loaded successfully!" << std::endl;
        std::cout << "Ready for attention computation." << std::endl;
        return true;
    }
    
    // Get a block of data for NoC transmission
    std::vector<float> getQueryBlock(int position, int block_id, int block_size = 64) {
        std::vector<float> block;
        int start_idx = block_id * block_size;
        int end_idx = std::min(start_idx + block_size, d_head);
        
        if (position < query_matrix.size()) {
            for (int i = start_idx; i < end_idx; i++) {
                block.push_back(query_matrix[position][i]);
            }
        }
        
        // Pad with zeros if needed
        while (block.size() < block_size) {
            block.push_back(0.0f);
        }
        
        return block;
    }
    
    std::vector<float> getKeyBlock(int position, int block_id, int block_size = 64) {
        std::vector<float> block;
        int start_idx = block_id * block_size;
        int end_idx = std::min(start_idx + block_size, d_head);
        
        if (position < key_matrix.size()) {
            for (int i = start_idx; i < end_idx; i++) {
                block.push_back(key_matrix[position][i]);
            }
        }
        
        // Pad with zeros if needed
        while (block.size() < block_size) {
            block.push_back(0.0f);
        }
        
        return block;
    }
};

// Example usage
int main() {
    LLaMAMatrixLoader loader;
    
    if (!loader.loadAllMatrices()) {
        std::cerr << "Failed to load matrices!" << std::endl;
        return 1;
    }
    
    // Example: Get first block of query data for position 0
    std::vector<float> q_block = loader.getQueryBlock(0, 0);
    std::cout << "\nFirst query block (position 0, block 0): ";
    for (int i = 0; i < 5; i++) {
        std::cout << q_block[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    // Example: Get first block of key data for position 0
    std::vector<float> k_block = loader.getKeyBlock(0, 0);
    std::cout << "First key block (position 0, block 0): ";
    for (int i = 0; i < 5; i++) {
        std::cout << k_block[i] << " ";
    }
    std::cout << "..." << std::endl;
    
    return 0;
}