#ifndef MC_MAPPING_HPP
#define MC_MAPPING_HPP

#include <cmath>
#include <algorithm>
#include <climits>
#include <iostream>
#include <iomanip>
#include <vector>


// 基于瓦片的映射：每个4x4瓦片内的PE映射到该瓦片的MC
inline int get_mc_for_pe(int ni_id, int x_num, int y_num) {
    int pe_x = ni_id / x_num;  // 行坐标
    int pe_y = ni_id % x_num;  // 列坐标
    
#if defined DATEMC2_4X4
    // 2个MC在4x4网格：简单的左右分区
    // MC在 (2,1) 和 (2,3)
    // 左半部分(y<2)映射到MC[0]，右半部分(y>=2)映射到MC[1]
    if (pe_y < 2) {
        return dest_list[0];  // 左侧MC at (2,1)
    } else {
        return dest_list[1];  // 右侧MC at (2,3)
    }
    
#elif defined DATEMC8_8X8
    // 8个MC在8x8网格：2x2瓦片，每个瓦片2个MC
    // 确定PE所在的瓦片
    int tile_row = pe_x / 4;  // 0 or 1
    int tile_col = pe_y / 4;  // 0 or 1
    int tile_id = tile_row * 2 + tile_col;  // 0-3
    
    // 每个瓦片内的局部坐标
    int local_x = pe_x % 4;
    int local_y = pe_y % 4;
    
    // 每个瓦片有2个MC，根据局部位置选择
    int base_mc_idx = tile_id * 2;
    if (local_y < 2) {
        return dest_list[base_mc_idx];      // 瓦片内左侧MC
    } else {
        return dest_list[base_mc_idx + 1];  // 瓦片内右侧MC
    }
    
#elif defined DATEMC32_16X16
    // 32个MC在16x16网格：4x4瓦片，每个瓦片2个MC
    int tile_row = pe_x / 4;  // 0-3
    int tile_col = pe_y / 4;  // 0-3
    int tile_id = tile_row * 4 + tile_col;  // 0-15
    
    // 每个瓦片内的局部坐标
    int local_x = pe_x % 4;
    int local_y = pe_y % 4;
    
    // 每个瓦片有2个MC，根据局部位置选择
    int base_mc_idx = tile_id * 2;
    if (local_y < 2) {
        return dest_list[base_mc_idx];      // 瓦片内左侧MC
    } else {
        return dest_list[base_mc_idx + 1];  // 瓦片内右侧MC
    }
    
#elif defined DATEMC128_32X32
    // 128个MC在32x32网格：8x8瓦片，每个瓦片2个MC
    int tile_row = pe_x / 4;  // 0-7
    int tile_col = pe_y / 4;  // 0-7
    int tile_id = tile_row * 8 + tile_col;  // 0-63
    
    // 每个瓦片内的局部坐标
    int local_x = pe_x % 4;
    int local_y = pe_y % 4;
    
    // 每个瓦片有2个MC，根据局部位置选择
    int base_mc_idx = tile_id * 2;
    if (local_y < 2) {
        return dest_list[base_mc_idx];      // 瓦片内左侧MC
    } else {
        return dest_list[base_mc_idx + 1];  // 瓦片内右侧MC
    }
    
#else
    // 默认返回第一个MC
    return dest_list[0];
#endif
}

// 验证函数：打印PE到MC的映射关系
inline void print_pe_to_mc_mapping(int grid_size, const int* dest_list, int mem_nodes) {
    std::cout << "\n=== PE to MC Mapping (" << grid_size << "x" << grid_size << " grid) ===" << std::endl;
    
    // 创建映射表
    std::vector<std::vector<int>> pe_mc_map(grid_size, std::vector<int>(grid_size, -1));
    
    // 计算每个PE的映射
    for (int x = 0; x < grid_size; x++) {
        for (int y = 0; y < grid_size; y++) {
            int ni_id = x * grid_size + y;
            int mc_id = get_mc_for_pe(ni_id, grid_size, grid_size);
            
            // 找到MC在dest_list中的索引
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
    
    // 打印映射表
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
            
            // 检查这个位置是否是MC
            bool is_mc = false;
            for (int i = 0; i < mem_nodes; i++) {
                if (dest_list[i] == (x * grid_size + y)) {
                    is_mc = true;
                    break;
                }
            }
            
            if (is_mc) {
                std::cout << " M";  // MC位置
            } else {
                std::cout << std::setw(2) << pe_mc_map[x][y];  // PE映射的MC索引
            }
        }
        std::cout << std::endl;
    }
    
    // 统计每个MC服务的PE数量
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
