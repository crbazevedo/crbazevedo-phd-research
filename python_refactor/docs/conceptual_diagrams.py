#!/usr/bin/env python3
"""
Conceptual Diagrams for Anticipatory SMS-EMOA Technical Documentation

This script generates various conceptual diagrams including:
1. Algorithm Flowchart
2. System Architecture
3. Anticipatory Learning Process
4. Rolling Horizon Framework
5. Mathematical Relationships
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, Circle, Arrow
import matplotlib.patches as mpatches

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_algorithm_flowchart():
    """Create the main algorithm flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    colors = {
        'start': '#4CAF50',
        'process': '#2196F3',
        'decision': '#FF9800',
        'data': '#9C27B0',
        'output': '#F44336'
    }
    
    # Create boxes
    boxes = [
        # Start
        {'pos': (5, 9), 'text': 'START\nInitialize Population', 'color': colors['start']},
        
        # Main loop
        {'pos': (5, 8), 'text': 'Generation t', 'color': colors['process']},
        
        # Selection
        {'pos': (2, 7), 'text': 'Tournament\nSelection', 'color': colors['process']},
        
        # Crossover & Mutation
        {'pos': (5, 7), 'text': 'Crossover &\nMutation', 'color': colors['process']},
        
        # Evaluation
        {'pos': (8, 7), 'text': 'Evaluate\nObjectives', 'color': colors['process']},
        
        # Anticipatory Learning
        {'pos': (5, 6), 'text': 'Anticipatory\nLearning', 'color': colors['data']},
        
        # Environmental Selection
        {'pos': (5, 5), 'text': 'Environmental\nSelection\n(Hypervolume)', 'color': colors['process']},
        
        # Pareto Front Update
        {'pos': (5, 4), 'text': 'Update\nPareto Front', 'color': colors['process']},
        
        # Store Stochastic Frontier
        {'pos': (8, 4), 'text': 'Store Stochastic\nPareto Frontier', 'color': colors['data']},
        
        # Decision
        {'pos': (5, 3), 'text': 't < G?', 'color': colors['decision']},
        
        # Output
        {'pos': (5, 1), 'text': 'Return\nPareto Front', 'color': colors['output']},
    ]
    
    # Draw boxes
    for box in boxes:
        x, y = box['pos']
        rect = FancyBboxPatch((x-1, y-0.5), 2, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=box['color'], 
                             edgecolor='black', 
                             linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, box['text'], ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    # Draw arrows
    arrows = [
        ((5, 8.5), (5, 7.5)),  # Generation to selection
        ((3, 7), (4, 7)),      # Selection to crossover
        ((6, 7), (7, 7)),      # Crossover to evaluation
        ((8, 6.5), (5, 6.5)),  # Evaluation to learning
        ((5, 5.5), (5, 4.5)),  # Learning to selection
        ((5, 3.5), (5, 2.5)),  # Selection to decision
        ((6, 3), (8, 3)),      # Decision to store
        ((8, 3.5), (8, 4.5)),  # Store to frontier
        ((4, 3), (4, 4)),      # Decision to update
        ((4, 4.5), (5, 4.5)),  # Update to selection
        ((5, 1.5), (5, 0.5)),  # Decision to output
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="black", linewidth=2)
        ax.add_patch(arrow)
    
    # Add loop arrow
    loop_arrow = ConnectionPatch((4, 3), (2, 6), "data", "data",
                                arrowstyle="->", shrinkA=5, shrinkB=5,
                                mutation_scale=20, fc="red", linewidth=3)
    ax.add_patch(loop_arrow)
    ax.text(1.5, 4.5, 'Loop', fontsize=12, fontweight='bold', color='red')
    
    plt.title('Anticipatory SMS-EMOA Algorithm Flowchart', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('algorithm_flowchart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_system_architecture():
    """Create system architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define components
    components = [
        # Data Layer
        {'pos': (2, 9), 'text': 'Market Data\n(Returns, Prices)', 'color': '#E3F2FD', 'type': 'data'},
        {'pos': (6, 9), 'text': 'Historical\nData', 'color': '#E3F2FD', 'type': 'data'},
        {'pos': (10, 9), 'text': 'Real-time\nFeeds', 'color': '#E3F2FD', 'type': 'data'},
        
        # Processing Layer
        {'pos': (2, 7), 'text': 'Data\nPreprocessing', 'color': '#F3E5F5', 'type': 'process'},
        {'pos': (6, 7), 'text': 'Statistics\nComputation', 'color': '#F3E5F5', 'type': 'process'},
        {'pos': (10, 7), 'text': 'Rolling\nWindow', 'color': '#F3E5F5', 'type': 'process'},
        
        # Algorithm Layer
        {'pos': (2, 5), 'text': 'SMS-EMOA\nOptimizer', 'color': '#E8F5E8', 'type': 'algorithm'},
        {'pos': (6, 5), 'text': 'Anticipatory\nLearning', 'color': '#E8F5E8', 'type': 'algorithm'},
        {'pos': (10, 5), 'text': 'Kalman\nFilter', 'color': '#E8F5E8', 'type': 'algorithm'},
        
        # Model Layer
        {'pos': (2, 3), 'text': 'Pareto\nFront', 'color': '#FFF3E0', 'type': 'model'},
        {'pos': (6, 3), 'text': 'Expected\nHypervolume', 'color': '#FFF3E0', 'type': 'model'},
        {'pos': (10, 3), 'text': 'Predictive\nModels', 'color': '#FFF3E0', 'type': 'model'},
        
        # Output Layer
        {'pos': (4, 1), 'text': 'Portfolio\nWeights', 'color': '#FFEBEE', 'type': 'output'},
        {'pos': (8, 1), 'text': 'Performance\nMetrics', 'color': '#FFEBEE', 'type': 'output'},
    ]
    
    # Draw components
    for comp in components:
        x, y = comp['pos']
        rect = FancyBboxPatch((x-1.5, y-0.8), 3, 1.6, 
                             boxstyle="round,pad=0.1", 
                             facecolor=comp['color'], 
                             edgecolor='black', 
                             linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, comp['text'], ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Add layer labels
    layers = [
        (1, 9.5, 'Data Layer'),
        (1, 7.5, 'Processing Layer'),
        (1, 5.5, 'Algorithm Layer'),
        (1, 3.5, 'Model Layer'),
        (1, 1.5, 'Output Layer'),
    ]
    
    for x, y, label in layers:
        ax.text(x, y, label, fontsize=12, fontweight='bold', rotation=90)
    
    # Add connections
    connections = [
        # Data to Processing
        ((2, 8.2), (2, 7.8)), ((6, 8.2), (6, 7.8)), ((10, 8.2), (10, 7.8)),
        # Processing to Algorithm
        ((2, 6.2), (2, 5.8)), ((6, 6.2), (6, 5.8)), ((10, 6.2), (10, 5.8)),
        # Algorithm to Model
        ((2, 4.2), (2, 3.8)), ((6, 4.2), (6, 3.8)), ((10, 4.2), (10, 3.8)),
        # Model to Output
        ((2, 2.2), (4, 1.8)), ((6, 2.2), (8, 1.8)), ((10, 2.2), (8, 1.8)),
        # Cross connections
        ((4, 7), (6, 7)), ((6, 5), (10, 5)), ((2, 5), (6, 5)),
    ]
    
    for start, end in connections:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=15, fc="gray", linewidth=1.5)
        ax.add_patch(arrow)
    
    plt.title('Anticipatory SMS-EMOA System Architecture', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_anticipatory_learning_diagram():
    """Create anticipatory learning process diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define process steps
    steps = [
        {'pos': (2, 9), 'text': 'Current\nState xₜ', 'color': '#E3F2FD'},
        {'pos': (5, 9), 'text': 'Monte Carlo\nSimulation', 'color': '#F3E5F5'},
        {'pos': (8, 9), 'text': 'Future States\nxₜ₊₁⁽ⁱ⁾', 'color': '#E8F5E8'},
        
        {'pos': (2, 7), 'text': 'Kalman\nPrediction', 'color': '#FFF3E0'},
        {'pos': (5, 7), 'text': 'Anticipative\nDistribution', 'color': '#FFEBEE'},
        {'pos': (8, 7), 'text': 'Expected\nHypervolume', 'color': '#F1F8E9'},
        
        {'pos': (2, 5), 'text': 'Prediction\nError ε', 'color': '#FFF8E1'},
        {'pos': (5, 5), 'text': 'Adaptive\nLearning Rate α', 'color': '#E0F2F1'},
        {'pos': (8, 5), 'text': 'Solution\nUpdate', 'color': '#FCE4EC'},
        
        {'pos': (2, 3), 'text': 'Non-dominance\nProbability', 'color': '#E8EAF6'},
        {'pos': (5, 3), 'text': 'Dirichlet\nMAP Filtering', 'color': '#E0F7FA'},
        {'pos': (8, 3), 'text': 'Updated\nSolution sₜ₊₁', 'color': '#F3E5F5'},
        
        {'pos': (5, 1), 'text': 'Learning\nEvent Log', 'color': '#FFCDD2'},
    ]
    
    # Draw steps
    for step in steps:
        x, y = step['pos']
        rect = FancyBboxPatch((x-1.2, y-0.6), 2.4, 1.2, 
                             boxstyle="round,pad=0.1", 
                             facecolor=step['color'], 
                             edgecolor='black', 
                             linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, step['text'], ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # Add flow arrows
    flows = [
        # Row 1
        ((3.2, 9), (4.8, 9)), ((6.2, 9), (7.8, 9)),
        # Row 2
        ((3.2, 7), (4.8, 7)), ((6.2, 7), (7.8, 7)),
        # Row 3
        ((3.2, 5), (4.8, 5)), ((6.2, 5), (7.8, 5)),
        # Row 4
        ((3.2, 3), (4.8, 3)), ((6.2, 3), (7.8, 3)),
        # Vertical connections
        ((2, 8.4), (2, 7.6)), ((5, 8.4), (5, 7.6)), ((8, 8.4), (8, 7.6)),
        ((2, 6.4), (2, 5.6)), ((5, 6.4), (5, 5.6)), ((8, 6.4), (8, 5.6)),
        ((2, 4.4), (2, 3.6)), ((5, 4.4), (5, 3.6)), ((8, 4.4), (8, 3.6)),
        # To output
        ((2, 2.4), (4.8, 1.6)), ((5, 2.4), (5, 1.6)), ((8, 2.4), (5.2, 1.6)),
    ]
    
    for start, end in flows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=15, fc="blue", linewidth=1.5)
        ax.add_patch(arrow)
    
    # Add mathematical formulas
    formulas = [
        (1, 8.5, 'xₜ = [ROIₜ, Riskₜ]ᵀ'),
        (4, 8.5, 'xₜ₊₁⁽ⁱ⁾ ~ p(xₜ₊₁|xₜ)'),
        (7, 8.5, 'μₐ = E[xₜ₊₁|xₜ]'),
        (1, 6.5, 'xₜ₊₁ = Fₜxₜ + wₜ'),
        (4, 6.5, 'p(xₜ₊₁|xₜ) = ∫ p(xₜ₊₁|xₜ,θ)p(θ|xₜ)dθ'),
        (7, 6.5, 'E[HV(Sₜ₊₁)]'),
        (1, 4.5, 'ε = ||xₜ - x̂ₜ||'),
        (4, 4.5, 'α = α₀ × (1+ε_K)⁻¹ × (1-H(p_dom))'),
        (7, 4.5, 'sₜ₊₁ = sₜ + α(μₐ - sₜ)'),
    ]
    
    for x, y, formula in formulas:
        ax.text(x, y, formula, fontsize=8, style='italic')
    
    plt.title('Anticipatory Learning Process', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('anticipatory_learning.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_rolling_horizon_diagram():
    """Create rolling horizon framework diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Timeline
    timeline_length = 10
    timeline_height = 0.5
    
    # Draw timeline
    timeline = Rectangle((1, 2.5), timeline_length, timeline_height, 
                        facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(timeline)
    
    # Add time markers
    for i in range(11):
        x = 1 + i * timeline_length / 10
        ax.plot([x, x], [2.5, 2.3], 'k-', linewidth=2)
        if i % 2 == 0:
            ax.text(x, 2.1, f't+{i*10}', ha='center', fontsize=10)
    
    # Define periods
    periods = [
        {'start': 1, 'end': 3, 'y': 4.5, 'label': 'Period 1\nTraining: 50d\nHolding: 30d'},
        {'start': 3, 'end': 5, 'y': 4.5, 'label': 'Period 2\nTraining: 50d\nHolding: 30d'},
        {'start': 5, 'end': 7, 'y': 4.5, 'label': 'Period 3\nTraining: 50d\nHolding: 30d'},
        {'start': 7, 'end': 9, 'y': 4.5, 'label': 'Period 4\nTraining: 50d\nHolding: 30d'},
    ]
    
    colors = ['#E3F2FD', '#F3E5F5', '#E8F5E8', '#FFF3E0']
    
    # Draw periods
    for i, period in enumerate(periods):
        # Training window
        train_rect = Rectangle((period['start'], period['y']), 1.5, 0.8, 
                              facecolor=colors[i], edgecolor='black', linewidth=2)
        ax.add_patch(train_rect)
        ax.text(period['start'] + 0.75, period['y'] + 0.4, 'Training\n50d', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Holding window
        hold_rect = Rectangle((period['start'] + 1.5, period['y']), 0.5, 0.8, 
                             facecolor='#FFCDD2', edgecolor='black', linewidth=2)
        ax.add_patch(hold_rect)
        ax.text(period['start'] + 1.75, period['y'] + 0.4, 'Hold\n30d', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Period label
        ax.text(period['start'] + 1, period['y'] - 0.3, period['label'], 
                ha='center', va='top', fontsize=8)
        
        # Connection to timeline
        ax.plot([period['start'] + 1, period['start'] + 1], [period['y'], 3.3], 
                'k--', alpha=0.5)
    
    # Add legend
    legend_elements = [
        patches.Patch(color='lightgray', label='Timeline'),
        patches.Patch(color='#E3F2FD', label='Training Window (50 days)'),
        patches.Patch(color='#FFCDD2', label='Holding Period (30 days)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add process flow
    process_steps = [
        (0.5, 1.5, '1. Extract\nTraining Data'),
        (2.5, 1.5, '2. Run\nSMS-EMOA'),
        (4.5, 1.5, '3. Hold\nPortfolios'),
        (6.5, 1.5, '4. Evaluate\nPerformance'),
        (8.5, 1.5, '5. Update\nWindow'),
    ]
    
    for x, y, step in process_steps:
        circle = Circle((x, y), 0.3, facecolor='#4CAF50', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, step, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    # Connect process steps
    for i in range(4):
        x1 = 0.8 + i * 2
        x2 = 2.2 + i * 2
        ax.arrow(x1, 1.5, x2-x1-0.4, 0, head_width=0.1, head_length=0.1, 
                fc='black', ec='black', linewidth=2)
    
    plt.title('Rolling Horizon Framework', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('rolling_horizon.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_mathematical_relationships():
    """Create mathematical relationships diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define mathematical components
    components = [
        # State Space
        {'pos': (2, 9), 'text': 'State Vector\nxₜ = [ROIₜ, Riskₜ, ΔROIₜ, ΔRiskₜ]ᵀ', 'color': '#E3F2FD'},
        
        # Kalman Filter
        {'pos': (5, 9), 'text': 'Kalman Filter\nxₜ₊₁ = Fₜxₜ + wₜ', 'color': '#F3E5F5'},
        
        # Anticipative Distribution
        {'pos': (8, 9), 'text': 'Anticipative Distribution\np(xₜ₊₁|xₜ)', 'color': '#E8F5E8'},
        
        # Hypervolume
        {'pos': (2, 7), 'text': 'Hypervolume\nHV(S, r) = λ(⋃ᵢ [sᵢ, r])', 'color': '#FFF3E0'},
        
        # Expected Hypervolume
        {'pos': (5, 7), 'text': 'Expected Future Hypervolume\nE[HV(Sₜ₊₁)]', 'color': '#FFEBEE'},
        
        # Adaptive Learning
        {'pos': (8, 7), 'text': 'Adaptive Learning Rate\nαₜ = α₀ × (1+ε_K)⁻¹ × (1-H(p_dom))', 'color': '#F1F8E9'},
        
        # Portfolio Optimization
        {'pos': (2, 5), 'text': 'Portfolio Optimization\nmin f₁(x) = -E[R(x)]\nmin f₂(x) = σ(x)', 'color': '#FFF8E1'},
        
        # Pareto Dominance
        {'pos': (5, 5), 'text': 'Pareto Dominance\n∀i: fᵢ(a) ≤ fᵢ(b) ∧ ∃j: fⱼ(a) < fⱼ(b)', 'color': '#E0F2F1'},
        
        # Dirichlet MAP
        {'pos': (8, 5), 'text': 'Dirichlet MAP\nθ_MAP = (α₀ + x - 1)/(∑α₀ᵢ + ∑xᵢ - K)', 'color': '#FCE4EC'},
        
        # Risk Calculation
        {'pos': (2, 3), 'text': 'Risk Calculation\nσ(x) = √(xᵀΣx)', 'color': '#E8EAF6'},
        
        # Return Calculation
        {'pos': (5, 3), 'text': 'Return Calculation\nR(x) = xᵀμ', 'color': '#E0F7FA'},
        
        # Solution Update
        {'pos': (8, 3), 'text': 'Solution Update\nsₜ₊₁ = sₜ + α(μₐ - sₜ)', 'color': '#F3E5F5'},
        
        # Performance Metrics
        {'pos': (5, 1), 'text': 'Performance Metrics\nSharpe Ratio, VaR, CVaR', 'color': '#FFCDD2'},
    ]
    
    # Draw components
    for comp in components:
        x, y = comp['pos']
        rect = FancyBboxPatch((x-1.5, y-0.8), 3, 1.6, 
                             boxstyle="round,pad=0.1", 
                             facecolor=comp['color'], 
                             edgecolor='black', 
                             linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, comp['text'], ha='center', va='center', 
                fontsize=8, fontweight='bold')
    
    # Add connections
    connections = [
        # State to Kalman
        ((3.5, 9), (4.5, 9)),
        # Kalman to Anticipative
        ((6.5, 9), (7.5, 9)),
        # State to Hypervolume
        ((2, 8.2), (2, 7.8)),
        # Anticipative to Expected Hypervolume
        ((8, 8.2), (5, 7.8)),
        # Expected Hypervolume to Adaptive Learning
        ((6.5, 7), (7.5, 7)),
        # Portfolio to Pareto
        ((3.5, 5), (4.5, 5)),
        # Pareto to Dirichlet
        ((6.5, 5), (7.5, 5)),
        # Risk and Return to Performance
        ((2, 2.2), (4.5, 1.8)), ((5, 2.2), (5, 1.8)),
        # Solution Update to Performance
        ((8, 2.2), (6.5, 1.8)),
    ]
    
    for start, end in connections:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=15, fc="blue", linewidth=1.5)
        ax.add_patch(arrow)
    
    plt.title('Mathematical Relationships in Anticipatory SMS-EMOA', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mathematical_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Generate all conceptual diagrams."""
    print("🎨 Generating Conceptual Diagrams for Technical Documentation...")
    
    print("📊 Creating Algorithm Flowchart...")
    create_algorithm_flowchart()
    
    print("🏗️  Creating System Architecture...")
    create_system_architecture()
    
    print("🧠 Creating Anticipatory Learning Diagram...")
    create_anticipatory_learning_diagram()
    
    print("⏰ Creating Rolling Horizon Framework...")
    create_rolling_horizon_diagram()
    
    print("📐 Creating Mathematical Relationships...")
    create_mathematical_relationships()
    
    print("✅ All conceptual diagrams generated successfully!")
    print("📁 Files saved:")
    print("   - algorithm_flowchart.png")
    print("   - system_architecture.png")
    print("   - anticipatory_learning.png")
    print("   - rolling_horizon.png")
    print("   - mathematical_relationships.png")

if __name__ == "__main__":
    main() 