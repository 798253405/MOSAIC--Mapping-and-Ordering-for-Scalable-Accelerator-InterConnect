#!/usr/bin/env python3
"""
Color Scheme Preview for NoC Performance Visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def show_color_schemes():
    """Display all color scheme options for comparison"""

    # Define all color schemes
    schemes = {
        '1. Nature Publishing': {
            'Baseline': '#0173B2',
            'SAMOS': '#56B4E9',
            'Affiliated': '#009E73',
            'Separated': '#F0E442',
            'MOSAIC-1': '#E69F00',
            'MOSAIC-2': '#CC79A7'
        },
        '2. IEEE Standard': {
            'Baseline': '#1f77b4',
            'SAMOS': '#ff7f0e',
            'Affiliated': '#2ca02c',
            'Separated': '#d62728',
            'MOSAIC-1': '#9467bd',
            'MOSAIC-2': '#8c564b'
        },
        '3. Science/Cell': {
            'Baseline': '#003f5c',
            'SAMOS': '#2f4b7c',
            'Affiliated': '#665191',
            'Separated': '#a05195',
            'MOSAIC-1': '#d45087',
            'MOSAIC-2': '#ff7c43'
        },
        '4. Colorblind Safe': {
            'Baseline': '#E1BE6A',
            'SAMOS': '#40B0A6',
            'Affiliated': '#5D3A9B',
            'Separated': '#E66100',
            'MOSAIC-1': '#5B8FA8',
            'MOSAIC-2': '#D41159'
        },
        '5. Monochrome Blue': {
            'Baseline': '#08306b',
            'SAMOS': '#08519c',
            'Affiliated': '#2171b5',
            'Separated': '#4292c6',
            'MOSAIC-1': '#6baed6',
            'MOSAIC-2': '#9ecae1'
        },
        '6. ACM CHI': {
            'Baseline': '#4e79a7',
            'SAMOS': '#f28e2c',
            'Affiliated': '#e15759',
            'Separated': '#76b7b2',
            'MOSAIC-1': '#59a14f',
            'MOSAIC-2': '#af7aa1'
        },
        '7. Tableau 10': {
            'Baseline': '#4e79a7',
            'SAMOS': '#f28e2b',
            'Affiliated': '#e15759',
            'Separated': '#76b7b2',
            'MOSAIC-1': '#59a14f',
            'MOSAIC-2': '#edc948'
        },
        '8. Original (Current)': {
            'Baseline': '#FF6B6B',
            'SAMOS': '#4ECDC4',
            'Affiliated': '#45B7D1',
            'Separated': '#96CEB4',
            'MOSAIC-1': '#FFEAA7',
            'MOSAIC-2': '#FD79A8'
        }
    }

    strategies = ['Baseline', 'SAMOS', 'Affiliated', 'Separated', 'MOSAIC-1', 'MOSAIC-2']

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Color Scheme Options for NoC Performance Visualization', fontsize=14, fontweight='bold')

    # Create subplots for each scheme
    n_schemes = len(schemes)
    n_cols = 2
    n_rows = (n_schemes + n_cols - 1) // n_cols

    for idx, (scheme_name, colors) in enumerate(schemes.items(), 1):
        ax = plt.subplot(n_rows, n_cols, idx)

        # Create bar chart with the color scheme
        x_pos = np.arange(len(strategies))
        heights = [5, 4.5, 4, 3.5, 3, 2.5]  # Different heights for visual variety

        bars = ax.bar(x_pos, heights, color=[colors[s] for s in strategies],
                      edgecolor='black', linewidth=1.5, alpha=0.9)

        # Add strategy names
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 6)
        ax.set_ylabel('Sample Values', fontsize=8)

        # Add title with characteristics
        if 'Nature' in scheme_name:
            subtitle = '\n(Colorblind-friendly, High impact journals)'
        elif 'IEEE' in scheme_name:
            subtitle = '\n(Classic technical papers)'
        elif 'Science' in scheme_name:
            subtitle = '\n(Modern, gradient effect)'
        elif 'Colorblind' in scheme_name:
            subtitle = '\n(Maximum accessibility)'
        elif 'Monochrome' in scheme_name:
            subtitle = '\n(Best for B&W printing)'
        elif 'ACM' in scheme_name:
            subtitle = '\n(Modern tech conferences)'
        elif 'Tableau' in scheme_name:
            subtitle = '\n(Data visualization standard)'
        else:
            subtitle = '\n(Bright, high contrast)'

        ax.set_title(scheme_name + subtitle, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Add recommendation text
    fig.text(0.5, 0.02,
             'Recommendations: #1 Nature (top journals), #2 IEEE (technical papers), #5 Monochrome (B&W printing)',
             ha='center', fontsize=10, style='italic', color='darkred')

    # Save and show
    plt.savefig('color_scheme_preview.pdf', dpi=150, bbox_inches='tight')
    print("Color scheme preview saved to: color_scheme_preview.pdf")
    plt.show()


if __name__ == "__main__":
    show_color_schemes()