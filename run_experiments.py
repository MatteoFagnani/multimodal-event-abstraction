"""
Script principale per l'esecuzione replicabile degli esperimenti della tesi:
Stage 1: Validazione singola clip (Clip 0000) - Test A (VLM) vs Test B (Domain Rules)
Stage 2: Estrazione sequenziale multi-clip (0000 - 0005) e scoperta del processo
Stage 3: Process Mining su scala globale (Intero primo tempo, 6.451 eventi)
Stage 4: Benchmark e Dataset di Comparazione Zero-Shot vs Domain Rules (767 clip)

Uso da riga di comando:
    python run_experiments.py --stage all
    python run_experiments.py --stage 1
    python run_experiments.py --stage 2
    python run_experiments.py --stage 3
    python run_experiments.py --stage 4
"""

import os
import sys
import argparse
import json
import pandas as pd
from typing import Dict, Any

from src.data_loader import load_sensor_data, load_ball_track, filter_active_pitch_players
from src.domain_rules import evaluate_action_with_domain_rules
from src.multi_clip_engine import run_multi_clip_pipeline
from src.full_match_engine import load_or_process_full_match_log, extract_match_analytics
from src.process_mining_analysis import discover_directly_follows_graph, plot_pass_network
from src.benchmark_comparison import run_full_match_benchmark

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def run_stage_1_single_clip():
    print("================================================================================")
    print("STAGE 1: VALIDAZIONE SINGOLA CLIP (CLIP 0000 - CALCIO D'INIZIO)")
    print("================================================================================")
    
    ball_path = os.path.join(DATA_DIR, "ball_tracks", "ball_track_0000.txt")
    sensor_path = os.path.join(DATA_DIR, "sensors", "sensors_clip0000.csv")
    
    ball_df = load_ball_track(ball_path)
    sensor_df = load_sensor_data(sensor_path)
    active_df = filter_active_pitch_players(sensor_df)
    
    print(f"Campionamenti ball tracking: {len(ball_df)} frame")
    print(f"Campionamenti sensori ZXY:   {len(active_df)} righe (attivi sul terreno)")
    
    # Ipotesi del Vision-Language Model (Test A - Zero-Shot Baseline)
    vlm_hypothesis = "Kickoff_and_Pass"
    
    # Applicazione regole di dominio (Test B - Neuro-Symbolic Validation)
    result = evaluate_action_with_domain_rules(
        primary_actor_id=5,
        vlm_hypothesis=vlm_hypothesis,
        ball_df=ball_df,
        sensor_df=active_df,
        clip_id="0000"
    )
    
    print("\nEsito della valutazione neuro-simbolica:")
    print(f"  - Attore primario:            Player_{result['actor_id']}")
    print(f"  - Ipotesi VLM (Test A):       {result['vlm_hypothesis']}")
    print(f"  - Attivita' validata (Test B): {result['validated_activity']}")
    print(f"  - Stato di validazione:       {result['decision_status']}")
    print(f"  - Spostamento palla:          {result['ball_displacement_px']} px")
    print(f"  - Distanza dal dischetto:     {result['player_dist_center_m']} m")
    print(f"  - Velocita' max calciatore:   {result['player_max_speed_kmh']} km/h")
    print("  - Regole applicate:")
    for r in result['rules_applied']:
        print(f"      * {r}")
        
    return result

def run_stage_2_multi_clip():
    print("\n================================================================================")
    print("STAGE 2: ESTRAZIONE SEQUENZIALE MULTI-CLIP (0000 - 0005, 18 SECONDI)")
    print("================================================================================")
    
    df_events = run_multi_clip_pipeline(DATA_DIR)
    out_csv = os.path.join(DATA_DIR, "event_logs", "event_log_multi_clip.csv")
    df_events.to_csv(out_csv, index=False)
    
    print(f"Event Log generato con successo: {len(df_events)} eventi totali.")
    print("\nSequenza di possesso palla (Traccia 'Possession_Phase_1'):")
    ball_trace = df_events[df_events['entity_type'] == 'ball']
    for idx, row in ball_trace.iterrows():
        print(f"  Clip {row['clip_id']} [{row['time:timestamp']}] | {row['org:resource']} -> {row['concept:name']} ({row['decision_status']})")
        
    return df_events

def run_stage_3_full_match():
    print("\n================================================================================")
    print("STAGE 3: PROCESS MINING SU SCALA GLOBALE (INTERO PRIMO TEMPO, 6.451 EVENTI)")
    print("================================================================================")
    
    full_log = load_or_process_full_match_log(DATA_DIR)
    analytics = extract_match_analytics(full_log, results_dir=RESULTS_DIR)
    
    print(f"Event log caricato: {analytics['total_events']:,} eventi su {analytics['unique_clips']} clip ({analytics['unique_clips']*3/60:.1f} minuti).")
    print(f"  - Eventi di gioco con palla:   {analytics['ball_action_events']:,}")
    print(f"  - Eventi cinematici giocatori: {analytics['player_kinematic_events']:,}")
    
    print("\nDistribuzione delle attivita' con la palla (Domain Knowledge Abstraction):")
    for act, cnt in analytics['ball_activity_distribution'].items():
        pct = (cnt / analytics['ball_action_events']) * 100.0
        print(f"  - {act:<25}: {cnt:>4} ({pct:5.1f}%)")
        
    print("\nIndicatori prestazionali dei calciatori (estratto):")
    for pid, stats in sorted(analytics['player_performance'].items())[:5]:
        print(f"  - {pid:<10}: Distanza={stats['total_displacement_km']:4.2f} km | Max Speed={stats['max_speed_kmh']:4.1f} km/h | Sprint={stats['sprint_count']:>2}")
        
    # Salvataggio delle statistiche aggregate in results
    stats_path = os.path.join(RESULTS_DIR, "match_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=4)
    print(f"\nStatistiche esportate in: {stats_path}")
    
    # Process Discovery DFG
    ball_events = full_log[full_log['entity_type'] == 'ball'].copy()
    try:
        import pm4py
        dfg, sa, ea = discover_directly_follows_graph(ball_events)
        print(f"\nGrafo delle dipendenze direttamente successive (DFG): {len(dfg)} transizioni identificate.")
        sorted_dfg = sorted(dfg.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 transizioni piu' frequenti:")
        for (a1, a2), freq in sorted_dfg:
            print(f"  - {a1} --> {a2}: {freq} occorrenze")
    except Exception as e:
        print(f"Nota su PM4Py: {e}")
        
    return full_log

def run_stage_4_comparison():
    print("\n================================================================================")
    print("STAGE 4: BENCHMARK GLOBALE E DATASET DI COMPARAZIONE (ZERO-SHOT VS DOMAIN RULES)")
    print("================================================================================")
    
    track_dir = os.path.join(BASE_DIR, "..", "SecondTest", "Balltrack", "track")
    sensor_csv = os.path.join(BASE_DIR, "..", "SecondTest", "2013-11-28_tromso_tottenham.csv")
    
    # Se eseguito all'interno della cartella repository con percorsi relativi
    if not os.path.exists(track_dir):
        track_dir = os.path.join(DATA_DIR, "ball_tracks")
        
    out_data = os.path.join(DATA_DIR, "event_logs")
    
    if os.path.exists(sensor_csv) and os.path.exists(track_dir) and len(os.listdir(track_dir)) > 10:
        df_comp, metrics = run_full_match_benchmark(track_dir, sensor_csv, out_data, RESULTS_DIR)
    else:
        # Carica il dataset di comparazione pre-generato
        comp_csv = os.path.join(out_data, "comparison_dataset_zero_shot_vs_domain_rules.csv")
        metrics_json = os.path.join(RESULTS_DIR, "comparison_summary_metrics.json")
        df_comp = pd.read_csv(comp_csv)
        with open(metrics_json, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    print(f"Dataset di comparazione caricato: {len(df_comp)} clip valutate.")
    print(f"Tasso di Concordanza Globale (Agreement Rate): {metrics['overall_agreement_rate_pct']:.2f}%")
    print(f"Clip Concordanti:   {metrics['concordant_clips']} ({metrics['concordant_clips']/metrics['total_evaluated_clips']*100:.1f}%)")
    print(f"Clip con Anomalie:  {metrics['discordant_clips']} ({metrics['discordant_clips']/metrics['total_evaluated_clips']*100:.1f}%)")
    
    print("\nTassonomia delle Discrepanze corrette dalle Regole di Dominio:")
    for cat, count in metrics['discrepancy_taxonomy'].items():
        if cat != 'NONE_CONCORDANT':
            pct = (count / metrics['total_evaluated_clips']) * 100.0
            print(f"  - {cat:<28}: {count:>4} clip ({pct:5.2f}%)")
            
    print("\nImpatto Diretto sulla Qualita' del Process Mining:")
    for imp, count in metrics['process_mining_impact_distribution'].items():
        pct = (count / metrics['total_evaluated_clips']) * 100.0
        print(f"  - {imp:<32}: {count:>4} clip ({pct:5.2f}%)")
        
    print(f"\nFile esportati:")
    print(f"  - CSV Completo:       {os.path.join(out_data, 'comparison_dataset_zero_shot_vs_domain_rules.csv')}")
    print(f"  - Matrice Confusione: {os.path.join(RESULTS_DIR, 'confusion_matrix_comparison.csv')}")
    print(f"  - Tabella LaTeX:      {os.path.join(RESULTS_DIR, 'latex_table_comparison.tex')}")
    
    return df_comp

def main():
    parser = argparse.ArgumentParser(description="Pipeline Multimodale di Event Abstraction e Process Mining per il Calcio")
    parser.add_argument("--stage", type=str, choices=["1", "2", "3", "4", "all"], default="all",
                        help="Fase sperimentale: 1 (Single Clip), 2 (Multi Clip), 3 (Full Match), 4 (Benchmark Comparazione), all (Tutte)")
    args = parser.parse_args()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if args.stage in ["1", "all"]:
        run_stage_1_single_clip()
    if args.stage in ["2", "all"]:
        run_stage_2_multi_clip()
    if args.stage in ["3", "all"]:
        run_stage_3_full_match()
    if args.stage in ["4", "all"]:
        run_stage_4_comparison()
        
    print("\n================================================================================")
    print("ESECUZIONE COMPLETATA CON SUCCESSO.")
    print(f"File e dataset disponibili nella cartella: {RESULTS_DIR} e {DATA_DIR}")
    print("================================================================================")

if __name__ == "__main__":
    main()
