#!/usr/bin/env python3
"""
Generate LLaMA-7B scale matrices for testing
LLaMA-7B parameters:
- Hidden dimension: 4096
- Number of heads: 32
- Head dimension: 128
- Sequence length: 2048
"""

import numpy as np
import os

def generate_llama_matrices(output_dir="llama_matrices"):
    """
    Generate Query and Key matrices for LLaMA-7B attention
    Each head processes 128-dimensional vectors
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # LLaMA-7B dimensions
    seq_length = 2048  # Sequence length
    head_dim = 128     # Dimension per attention head
    
    print(f"Generating LLaMA-7B attention matrices...")
    print(f"Sequence length: {seq_length}")
    print(f"Head dimension: {head_dim}")
    print(f"Matrix shape: {seq_length} x {head_dim}")
    
    # Generate Query matrix (seq_length x head_dim)
    # Use normal distribution with small values to simulate real attention weights
    query_matrix = np.random.randn(seq_length, head_dim).astype(np.float32) * 0.1
    
    # Generate Key matrix (seq_length x head_dim)
    key_matrix = np.random.randn(seq_length, head_dim).astype(np.float32) * 0.1
    
    # Add some structure to make it more realistic
    # Real attention often has some patterns
    for i in range(0, seq_length, 64):
        # Add slight correlation in blocks
        query_matrix[i:i+64] += np.random.randn(1, head_dim).astype(np.float32) * 0.05
        key_matrix[i:i+64] += np.random.randn(1, head_dim).astype(np.float32) * 0.05
    
    # Save Query matrix
    query_path = os.path.join(output_dir, "llama_query_2048x128.txt")
    with open(query_path, 'w') as f:
        f.write(f"{seq_length} {head_dim}\n")  # Write dimensions
        for row in query_matrix:
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
    print(f"Saved Query matrix to {query_path}")
    
    # Save Key matrix  
    key_path = os.path.join(output_dir, "llama_key_2048x128.txt")
    with open(key_path, 'w') as f:
        f.write(f"{seq_length} {head_dim}\n")  # Write dimensions
        for row in key_matrix:
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
    print(f"Saved Key matrix to {key_path}")
    
    # Generate smaller test matrices for quick validation (512x128)
    test_seq_length = 512
    query_test = np.random.randn(test_seq_length, head_dim).astype(np.float32) * 0.1
    key_test = np.random.randn(test_seq_length, head_dim).astype(np.float32) * 0.1
    
    # Save test Query matrix
    query_test_path = os.path.join(output_dir, "llama_query_512x128.txt")
    with open(query_test_path, 'w') as f:
        f.write(f"{test_seq_length} {head_dim}\n")
        for row in query_test:
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
    print(f"Saved test Query matrix to {query_test_path}")
    
    # Save test Key matrix
    key_test_path = os.path.join(output_dir, "llama_key_512x128.txt")
    with open(key_test_path, 'w') as f:
        f.write(f"{test_seq_length} {head_dim}\n")
        for row in key_test:
            f.write(" ".join(f"{val:.6f}" for val in row) + "\n")
    print(f"Saved test Key matrix to {key_test_path}")
    
    # Print statistics
    print("\nMatrix statistics:")
    print(f"Query matrix - min: {query_matrix.min():.4f}, max: {query_matrix.max():.4f}, mean: {query_matrix.mean():.4f}")
    print(f"Key matrix - min: {key_matrix.min():.4f}, max: {key_matrix.max():.4f}, mean: {key_matrix.mean():.4f}")
    
    # Calculate expected attention scores (Q @ K^T for first few elements)
    sample_attention = query_matrix[:8] @ key_matrix[:8].T
    print(f"\nSample attention scores (8x8 block):")
    print(f"Min: {sample_attention.min():.4f}, Max: {sample_attention.max():.4f}, Mean: {sample_attention.mean():.4f}")
    
    print("\nGeneration complete!")
    return query_path, key_path

if __name__ == "__main__":
    generate_llama_matrices()