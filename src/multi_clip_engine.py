"""
Modulo per l'esecuzione della pipeline sequenziale multi-clip (0000 - 0005)
per la ricostruzione della fase iniziale di possesso palla (18 secondi).
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .data_loader import load_sensor_data, load_ball_track
from .domain_rules import evaluate_action_with_domain_rules

def run_multi_clip_pipeline(data_dir: str) -> pd.DataFrame:
    """
    Esegue l'astrazione di eventi su 6 clip consecutive (0000 - 0005).
    Ritorna un DataFrame strutturato come Event Log conforme agli standard IEEE XES / Process Mining.
    """
    sensor_path = os.path.join(data_dir, "sensors", "sensors_multi_clip.csv")
    if not os.path.exists(sensor_path):
        raise FileNotFoundError(f"File sensori multi-clip non trovato: {sensor_path}")
        
    sensors_all = load_sensor_data(sensor_path)
    
    # Mappatura ground truth dei passaggi e attori per le 6 clip iniziali
    clip_metadata = [
        {"clip": "0000", "ts_start": "2013-11-28 19:03:54", "ts_end": "2013-11-28 19:03:57", "actor": 5, "hypo": "Kickoff_and_Pass"},
        {"clip": "0001", "ts_start": "2013-11-28 19:03:57", "ts_end": "2013-11-28 19:04:00", "actor": 12, "hypo": "Ball_Control_and_Carry"},
        {"clip": "0002", "ts_start": "2013-11-28 19:04:00", "ts_end": "2013-11-28 19:04:03", "actor": 11, "hypo": "Backward_Pass"},
        {"clip": "0003", "ts_start": "2013-11-28 19:04:03", "ts_end": "2013-11-28 19:04:06", "actor": 2, "hypo": "Backward_Pass"},
        {"clip": "0004", "ts_start": "2013-11-28 19:04:06", "ts_end": "2013-11-28 19:04:09", "actor": 4, "hypo": "Long_Forward_Pass"},
        {"clip": "0005", "ts_start": "2013-11-28 19:04:09", "ts_end": "2013-11-28 19:04:12", "actor": 14, "hypo": "Backward_Pass"}
    ]
    
    events = []
    
    for meta in clip_metadata:
        c_id = meta["clip"]
        ball_path = os.path.join(data_dir, "ball_tracks", f"ball_track_{c_id}.txt")
        if not os.path.exists(ball_path):
            continue
            
        b_df = load_ball_track(ball_path)
        s_slice = sensors_all[(sensors_all['ts'] >= meta["ts_start"]) & (sensors_all['ts'] <= meta["ts_end"])].copy()
        
        # Validazione con regole di dominio
        res = evaluate_action_with_domain_rules(
            primary_actor_id=meta["actor"],
            vlm_hypothesis=meta["hypo"],
            ball_df=b_df,
            sensor_df=s_slice,
            clip_id=c_id
        )
        
        # Genera evento per l'azione della palla (traccia 'Ball_Possession_Phase')
        events.append({
            'case:concept:name': 'Possession_Phase_1',
            'concept:name': res['validated_activity'],
            'time:timestamp': meta['ts_start'],
            'org:resource': f"Player_{res['actor_id']}",
            'clip_id': c_id,
            'entity_type': 'ball',
            'displacement_px': res['ball_displacement_px'],
            'decision_status': res['decision_status']
        })
        
        # Genera eventi per i calciatori attivi (corsa, sprint, posizionamento)
        for pid in sorted(s_slice['player_id'].unique()):
            pdf = s_slice[s_slice['player_id'] == pid]
            if pdf.empty:
                continue
            max_spd = float(pdf['speed_kmh'].max())
            mean_spd = float(pdf['speed_kmh'].mean())
            disp_m = float(np.hypot(pdf.iloc[-1]['x'] - pdf.iloc[0]['x'], pdf.iloc[-1]['y'] - pdf.iloc[0]['y']))
            
            if max_spd >= 24.0:
                act = "High_Speed_Sprint"
            elif max_spd >= 14.0:
                act = "Running_Support"
            elif disp_m >= 3.0:
                act = "Tactical_Repositioning"
            else:
                act = "Standing_Marking"
                
            events.append({
                'case:concept:name': f"Player_{pid}_Trace",
                'concept:name': act,
                'time:timestamp': meta['ts_start'],
                'org:resource': f"Player_{pid}",
                'clip_id': c_id,
                'entity_type': 'player',
                'speed_max_kmh': round(max_spd, 1),
                'speed_mean_kmh': round(mean_spd, 1),
                'displacement_m': round(disp_m, 2),
                'decision_status': 'TELEMETRY_CONFIRMED'
            })
            
    df_events = pd.DataFrame(events)
    return df_events
