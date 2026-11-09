"""
Modulo per l'acquisizione e il preprocessamento dei flussi di dati multimodali:
- Telemetria sensori ZXY a 20Hz (posizione x,y, velocita', accelerazione)
- Tracciamento ground truth della palla (frame-by-frame)
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from .config import PITCH_WIDTH_M

def load_sensor_data(filepath: str) -> pd.DataFrame:
    """
    Carica i dati di telemetria dei sensori corporei ZXY a 20Hz.
    Calcola la velocita' in km/h se non gia' presente.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File sensori non trovato: {filepath}")
    
    df = pd.read_csv(filepath)
    if 'speed_kmh' not in df.columns and 'speed' in df.columns:
        df['speed_kmh'] = df['speed'] * 3.6
        
    return df

def filter_active_pitch_players(sensor_df: pd.DataFrame, max_y: float = 65.0) -> pd.DataFrame:
    """
    Filtra i giocatori escludendo quelli posizionati in panchina (coordinata Y > 65m).
    """
    # Identifica i player_id la cui posizione iniziale e' sul terreno di gioco
    active_ids = []
    for pid in sensor_df['player_id'].unique():
        pdf = sensor_df[sensor_df['player_id'] == pid]
        if not pdf.empty:
            start_y = pdf.iloc[0]['y']
            if 0.0 <= start_y <= max_y:
                active_ids.append(pid)
                
    return sensor_df[sensor_df['player_id'].isin(active_ids)].copy()

def load_ball_track(filepath: str) -> pd.DataFrame:
    """
    Carica le coordinate del pallone da file di testo strutturato (frame x y).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File ball track non trovato: {filepath}")
        
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=['frame', 'x', 'y'])
    return df

def compute_kinematic_summary(player_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcola le metriche cinematiche di un giocatore durante una finestra temporale:
    spostamento totale (m), velocita' media (km/h), velocita' di picco (km/h).
    """
    if player_df.empty:
        return {'displacement_m': 0.0, 'mean_speed_kmh': 0.0, 'max_speed_kmh': 0.0}
        
    p_start = player_df.iloc[0]
    p_end = player_df.iloc[-1]
    
    disp = float(np.hypot(p_end['x'] - p_start['x'], p_end['y'] - p_start['y']))
    mean_s = float(player_df['speed_kmh'].mean())
    max_s = float(player_df['speed_kmh'].max())
    
    return {
        'displacement_m': round(disp, 2),
        'mean_speed_kmh': round(mean_s, 2),
        'max_speed_kmh': round(max_s, 2),
        'start_pos': (float(p_start['x']), float(p_start['y'])),
        'end_pos': (float(p_end['x']), float(p_end['y']))
    }
