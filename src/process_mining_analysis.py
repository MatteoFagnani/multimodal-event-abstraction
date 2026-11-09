"""
Modulo di Process Mining basato sulla libreria standard PM4Py e NetworkX.
Implementa:
- Scoperta del modello di processo tramite Directly-Follows Graph (DFG)
- Social Network Analysis (Handover of Work / Pass Network)
- Esportazione grafica conforme a pubblicazioni scientifiche
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional

def prepare_pm4py_event_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formatta il DataFrame garantendo la conformita' agli standard IEEE XES per PM4Py.
    """
    event_log = df.copy()
    event_log['time:timestamp'] = pd.to_datetime(event_log['time:timestamp'])
    event_log = event_log.sort_values(by='time:timestamp')
    
    # Formattazione PM4Py
    import pm4py
    formatted_log = pm4py.format_dataframe(
        event_log,
        case_id='case:concept:name',
        activity_key='concept:name',
        timestamp_key='time:timestamp'
    )
    return formatted_log

def discover_directly_follows_graph(ball_events_df: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], int], Dict[str, int], Dict[str, int]]:
    """
    Calcola le frequenze di transizione direttamente successive (DFG) per le azioni di gioco.
    """
    import pm4py
    df_pm4py = prepare_pm4py_event_log(ball_events_df)
    dfg, start_activities, end_activities = pm4py.discover_dfg(df_pm4py)
    return dfg, start_activities, end_activities

def plot_pass_network(
    pass_transitions: Dict[Tuple[str, str], int],
    output_path: str
) -> None:
    """
    Genera il grafo della rete dei passaggi (Social Network Analysis / Handover of Work)
    utilizzando NetworkX e Matplotlib.
    """
    G = nx.DiGraph()
    for (p1, p2), weight in pass_transitions.items():
        if p1 != p2 and weight > 0:
            G.add_edge(p1, p2, weight=weight)
            
    if len(G.nodes) == 0:
        return
        
    plt.figure(figsize=(10, 8), dpi=300)
    pos = nx.spring_layout(G, k=1.8, seed=42)
    
    # Pesi degli archi
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [1.0 + 4.0 * (w / max_w) for w in weights]
    
    # Grado in entrata/uscita come dimensione del nodo
    node_sizes = [1200 + 400 * G.degree(n) for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='#1E88E5', alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=11, font_color='white', font_weight='bold')
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='-|>',
        arrowsize=18,
        edge_color='#555555',
        width=edge_widths,
        connectionstyle='arc3,rad=0.15'
    )
    
    edge_labels = {(u, v): str(d['weight']) for u, v, d in G.edges(data=True) if d['weight'] >= 2}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='#D32F2F')
    
    plt.title("Rete dei Passaggi - Handover of Work (SNA)", fontsize=13, fontweight='bold', pad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
