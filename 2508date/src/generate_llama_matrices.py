#!/usr/bin/env python3
"""
Generate LLaMA-scale attention matrices for NoC simulation
LLaMA-7B parameters:
- Hidden dimension: 4096
- Number of heads: 32
- Head dimension: 128 (4096/32)
- Sequence length: 2048 (typical)
"""

import numpy as np
import sys
import os

# LLaMA-7B configuration
SEQUENCE_LENGTH = 2048  # Can be 2048 or 4096
D_HEAD = 128           # Head dimension (4096/32)
NUM_HEADS = 32         # Number of attention heads

# For NoC testing, we might want smaller sizes
# Uncomment below for smaller test size:
# SEQUENCE_LENGTH = 512  # Smaller for testing
# D_HEAD = 64           # Smaller for testing

def generate_attention_matrices(seq_len, d_head, seed=42):
    """
    Generate Q, K, V matrices for one attention head
    
    Args:
        seq_len: Sequence length (2048 for LLaMA)
        d_head: Head dimension (128 for LLaMA-7B)
        seed: Random seed for reproducibility
    
    Returns:
        query, key, value matrices
    """
    np.random.seed(seed)
    
    # Generate matrices with values in [-1, 1] range (similar to your current code)
    # In reality, these would come from layer normalization, so they're roughly normalized
    query = np.random.uniform(-1.0, 1.0, (seq_len, d_head)).astype(np.float32)
    key = np.random.uniform(-1.0, 1.0, (seq_len, d_head)).astype(np.float32)
    value = np.random.uniform(-1.0, 1.0, (seq_len, d_head)).astype(np.float32)
    
    return query, key, value

def save_matrices_for_cpp(query, key, value, output_dir="./Input/"):
    """
    Save matrices in format that C++ code can read
    Format: text files with space-separated values
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as text files for easy C++ reading
    np.savetxt(os.path.join(output_dir, "llama_query.txt"), query, fmt='%.6f')
    np.savetxt(os.path.join(output_dir, "llama_key.txt"), key, fmt='%.6f')
    np.savetxt(os.path.join(output_dir, "llama_value.txt"), value, fmt='%.6f')
    
    # Also save metadata
    with open(os.path.join(output_dir, "llama_metadata.txt"), 'w') as f:
        f.write(f"SEQUENCE_LENGTH {query.shape[0]}\n")
        f.write(f"D_HEAD {query.shape[1]}\n")
        f.write(f"MATRIX_SIZE {query.shape[0]} {query.shape[1]}\n")
    
    print(f"Saved matrices to {output_dir}")
    print(f"  Query shape: {query.shape}")
    print(f"  Key shape: {key.shape}")
    print(f"  Value shape: {value.shape}")

def save_attention_scores_example(query, key, output_dir="./Input/"):
    """
    Compute and save example attention scores (Q × K^T)
    This shows what the MAC units need to compute
    """
    # Compute attention scores for first few positions (too big to save all)
    # Q × K^T = [seq_len × d_head] × [d_head × seq_len] = [seq_len × seq_len]
    
    # For NoC simulation, we typically compute one row at a time
    # Each row represents attention from one position to all other positions
    
    # Example: compute first 10 rows of attention scores
    num_example_rows = min(10, query.shape[0])
    attention_scores = np.zeros((num_example_rows, key.shape[0]), dtype=np.float32)
    
    for i in range(num_example_rows):
        # Compute attention scores for position i
        # This is what each MAC unit computes: dot product of Q[i] with all K
        attention_scores[i] = np.dot(query[i], key.T)
    
    np.savetxt(os.path.join(output_dir, "llama_attention_scores_example.txt"), 
               attention_scores, fmt='%.6f')
    print(f"  Attention scores example shape: {attention_scores.shape}")

def generate_blocked_data_for_noc(query, key, block_size=64):
    """
    Generate blocked data suitable for NoC transmission
    Each block is what would be sent in one packet
    
    Args:
        block_size: Number of elements per block (64 matches your current code)
    """
    seq_len, d_head = query.shape
    
    # For attention computation, we need to send:
    # 1. Chunks of Q (for one position)
    # 2. Chunks of K (for multiple positions)
    
    blocks = []
    
    # Example: Process first position's attention
    # Need Q[0] (128 elements) and all K vectors
    
    # Split Q[0] into blocks of 64 elements
    q_blocks = []
    for i in range(0, d_head, block_size):
        end = min(i + block_size, d_head)
        q_block = query[0, i:end]
        if len(q_block) < block_size:
            # Pad with zeros if needed
            q_block = np.pad(q_block, (0, block_size - len(q_block)), 'constant')
        q_blocks.append(q_block)
    
    # For each K vector, also split into blocks
    # In practice, we'd send corresponding blocks of K for dot product
    
    print(f"\nBlocking example for NoC:")
    print(f"  Q[0] split into {len(q_blocks)} blocks of size {block_size}")
    print(f"  Each attention computation needs {seq_len} K vectors")
    print(f"  Total data per attention row: {seq_len * d_head} elements")
    
    return q_blocks

def main():
    print("=== Generating LLaMA-scale Attention Matrices ===")
    print(f"Configuration:")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Head Dimension: {D_HEAD}")
    print(f"  Total elements per matrix: {SEQUENCE_LENGTH * D_HEAD:,}")
    print()
    
    # Generate matrices
    query, key, value = generate_attention_matrices(SEQUENCE_LENGTH, D_HEAD)
    
    # Save for C++ code
    save_matrices_for_cpp(query, key, value)
    
    # Save example attention scores
    save_attention_scores_example(query, key)
    
    # Show blocking for NoC
    generate_blocked_data_for_noc(query, key)
    
    print("\n=== Matrix Generation Complete ===")
    print("\nFor C++ integration:")
    print("1. Read matrices from Input/llama_query.txt, llama_key.txt, llama_value.txt")
    print("2. Each line in the file is one row of the matrix")
    print("3. Parse space-separated float values")
    print("\nMemory requirements:")
    print(f"  Per matrix: {SEQUENCE_LENGTH * D_HEAD * 4 / (1024*1024):.2f} MB (float32)")
    print(f"  Total for Q+K+V: {3 * SEQUENCE_LENGTH * D_HEAD * 4 / (1024*1024):.2f} MB")
    print(f"  Attention scores (Q×K^T): {SEQUENCE_LENGTH * SEQUENCE_LENGTH * 4 / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()