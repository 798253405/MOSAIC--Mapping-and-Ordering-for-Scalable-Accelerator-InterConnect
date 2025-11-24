#ifndef MC_MAPPING_HPP
#define MC_MAPPING_HPP

#include <cmath>
#include <algorithm>
#include <climits>
#include <iostream>
#include <iomanip>
#include <vector>


// ：4x4PEMC
inline int get_mc_for_pe(int ni_id, int x_num, int y_num) {
    int pe_x = ni_id / x_num;
    int pe_y = ni_id % x_num;

#if defined NOCSIZEMC2_4X4
    // 2MC4x4：
    // MC (2,1)  (2,3)
    // (y<2)MC[0]，(y>=2)MC[1]
    if (pe_y < 2) {
        return dest_list[0];  // MC at (2,1)
    } else {
        return dest_list[1];  // MC at (2,3)
    }

#elif defined NOCSIZEMC8_8X8
    // 8MC8x8：2x2，2MC
    // PE
    int tile_row = pe_x / 4;  // 0 or 1
    int tile_col = pe_y / 4;  // 0 or 1
    int tile_id = tile_row * 2 + tile_col;  // 0-3

    int local_x = pe_x % 4;
    int local_y = pe_y % 4;

    // 2MC，
    int base_mc_idx = tile_id * 2;
    if (local_y < 2) {
        return dest_list[base_mc_idx];      // MC
    } else {
        return dest_list[base_mc_idx + 1];  // MC
    }

#elif defined NOCSIZEMC32_16X16
    // 32MC16x16：4x4，2MC
    int tile_row = pe_x / 4;  // 0-3
    int tile_col = pe_y / 4;  // 0-3
    int tile_id = tile_row * 4 + tile_col;  // 0-15

    int local_x = pe_x % 4;
    int local_y = pe_y % 4;

    // 2MC，
    int base_mc_idx = tile_id * 2;
    if (local_y < 2) {
        return dest_list[base_mc_idx];      // MC
    } else {
        return dest_list[base_mc_idx + 1];  // MC
    }

#elif defined NOCSIZEMC128_32X32
    // 128MC32x32：8x8，2MC
    int tile_row = pe_x / 4;  // 0-7
    int tile_col = pe_y / 4;  // 0-7
    int tile_id = tile_row * 8 + tile_col;  // 0-63

    int local_x = pe_x % 4;
    int local_y = pe_y % 4;

    // 2MC，
    int base_mc_idx = tile_id * 2;
    if (local_y < 2) {
        return dest_list[base_mc_idx];      // MC
    } else {
        return dest_list[base_mc_idx + 1];  // MC
    }

#else
    // MC
    return dest_list[0];
#endif
}

// ：PEMC
inline void print_pe_to_mc_mapping(int grid_size, const int* dest_list, int mem_nodes) {
    std::cout << "\n=== PE to MC Mapping (" << grid_size << "x" << grid_size << " grid) ===" << std::endl;

    std::vector<std::vector<int>> pe_mc_map(grid_size, std::vector<int>(grid_size, -1));

    // PE
    for (int x = 0; x < grid_size; x++) {
        for (int y = 0; y < grid_size; y++) {
            int ni_id = x * grid_size + y;
            int mc_id = get_mc_for_pe(ni_id, grid_size, grid_size);

            // MCdest_list
            int mc_idx = -1;
            for (int i = 0; i < mem_nodes; i++) {
                if (dest_list[i] == mc_id) {
                    mc_idx = i;
                    break;
                }
            }
            pe_mc_map[x][y] = mc_idx;
        }
    }

    std::cout << "MC Index mapping (shows which MC index each PE maps to):" << std::endl;
    std::cout << "    ";
    for (int y = 0; y < grid_size; y++) {
        if (y % 4 == 0 && y > 0) std::cout << " ";
        std::cout << std::setw(2) << y;
    }
    std::cout << std::endl;

    for (int x = 0; x < grid_size; x++) {
        if (x % 4 == 0 && x > 0) {
            std::cout << "    ";
            for (int y = 0; y < grid_size + grid_size/4 - 1; y++) {
                std::cout << "--";
            }
            std::cout << std::endl;
        }
        std::cout << std::setw(2) << x << ": ";
        for (int y = 0; y < grid_size; y++) {
            if (y % 4 == 0 && y > 0) std::cout << "|";

            // MC
            bool is_mc = false;
            for (int i = 0; i < mem_nodes; i++) {
                if (dest_list[i] == (x * grid_size + y)) {
                    is_mc = true;
                    break;
                }
            }

            if (is_mc) {
                std::cout << " M";  // MC
            } else {
                std::cout << std::setw(2) << pe_mc_map[x][y];  // PEMC
            }
        }
        std::cout << std::endl;
    }

    // MCPE
    std::vector<int> mc_pe_count(mem_nodes, 0);
    for (int x = 0; x < grid_size; x++) {
        for (int y = 0; y < grid_size; y++) {
            int mc_idx = pe_mc_map[x][y];
            if (mc_idx >= 0 && mc_idx < mem_nodes) {
                mc_pe_count[mc_idx]++;
            }
        }
    }

    std::cout << "\nMC Load Distribution:" << std::endl;
    for (int i = 0; i < mem_nodes; i++) {
        int mc_id = dest_list[i];
        int mc_x = mc_id / grid_size;
        int mc_y = mc_id % grid_size;
        std::cout << "MC[" << i << "] at (" << mc_x << "," << mc_y
                  << "): serves " << mc_pe_count[i] << " PEs" << std::endl;
    }
}

#endif // MC_MAPPING_HPP
