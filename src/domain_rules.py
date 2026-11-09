"""
Motore di Domain Knowledge Rules per la validazione neuro-simbolica
e l'Event Abstraction a partire da dati cinematici e spaziali.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .config import (
    CENTER_SPOT_M,
    CENTER_CIRCLE_RADIUS_M,
    BALL_STATIONARY_THRESH_PX,
    BALL_SHORT_PASS_THRESH_PX,
    BALL_LONG_PASS_THRESH_PX
)

def evaluate_action_with_domain_rules(
    primary_actor_id: int,
    vlm_hypothesis: str,
    ball_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    clip_id: str = "0000"
) -> Dict[str, Any]:
    """
    Applica il sistema di regole formali di dominio per validare o raffinare
    l'ipotesi generata dal modello di visione (VLM).
    """
    # 1. Cinematica della palla
    b_start = ball_df.iloc[0]
    b_end = ball_df.iloc[-1]
    dx = float(b_end['x'] - b_start['x'])
    dy = float(b_end['y'] - b_start['y'])
    ball_disp_px = float(np.hypot(dx, dy))
    
    # 2. Cinematica del calciatore attore primario
    pdf = sensor_df[sensor_df['player_id'] == primary_actor_id]
    if pdf.empty:
        p_start_pos = (0.0, 0.0)
        p_dist_center = 999.0
        p_disp_m = 0.0
        p_max_speed = 0.0
        p_mean_speed = 0.0
    else:
        p_s = pdf.iloc[0]
        p_e = pdf.iloc[-1]
        p_start_pos = (float(p_s['x']), float(p_s['y']))
        p_dist_center = float(np.hypot(p_s['x'] - CENTER_SPOT_M[0], p_s['y'] - CENTER_SPOT_M[1]))
        p_disp_m = float(np.hypot(p_e['x'] - p_s['x'], p_e['y'] - p_s['y']))
        p_max_speed = float(pdf['speed_kmh'].max())
        p_mean_speed = float(pdf['speed_kmh'].mean())

    rules_triggered = []
    
    # Regola 1: Vincolo regolamentare del calcio d'inizio
    # Se la clip e' all'avvio del tempo (t=0, clip 0000) e il calciatore e' all'interno del cerchio di centrocampo
    is_kickoff = (clip_id == "0000" and p_dist_center <= 5.0)
    if is_kickoff:
        rules_triggered.append(f"R1_KICKOFF_REGULATION: Giocatore a {p_dist_center:.2f}m dal dischetto (entro il cerchio centrale)")

    # Regola 2: Classificazione basata sullo spostamento del pallone
    if ball_disp_px <= BALL_STATIONARY_THRESH_PX:
        ball_dynamic = "Stationary_Play"
        rules_triggered.append(f"R2_BALL_STATIONARY: Spostamento palla {ball_disp_px:.1f}px <= {BALL_STATIONARY_THRESH_PX}px")
    elif ball_disp_px <= BALL_SHORT_PASS_THRESH_PX:
        ball_dynamic = "Ball_Control_and_Carry"
        rules_triggered.append(f"R2_BALL_CONTROL: Spostamento palla {ball_disp_px:.1f}px in conduzione ravvicinata")
    elif ball_disp_px > BALL_LONG_PASS_THRESH_PX:
        ball_dynamic = "Long_Forward_Pass" if dx > 0 else "Long_Clearance"
        rules_triggered.append(f"R2_BALL_LONG_TRAJECTORY: Rilancio lungo ({ball_disp_px:.1f}px, dx={dx:.1f}px)")
    else:
        ball_dynamic = "Forward_Pass" if dx > 100 else ("Backward_Pass" if dx < -100 else "Lateral_Pass")
        rules_triggered.append(f"R2_BALL_PASS: Traiettoria ({ball_disp_px:.1f}px, dx={dx:.1f}px) -> {ball_dynamic}")

    # Regola 3: Decisione di sintesi neuro-simbolica
    if is_kickoff and ball_disp_px > BALL_SHORT_PASS_THRESH_PX:
        validated_activity = "Kickoff_and_Pass"
        decision_status = "CONFIRMED_BY_KINEMATICS" if vlm_hypothesis == "Kickoff_and_Pass" else "REFINED_BY_RULES"
    elif ball_dynamic == "Ball_Control_and_Carry":
        validated_activity = "Ball_Control_and_Carry"
        decision_status = "CONFIRMED_BY_KINEMATICS" if "Control" in vlm_hypothesis or "Dribbling" in vlm_hypothesis else "REFINED_BY_RULES"
    else:
        validated_activity = ball_dynamic
        decision_status = "CONFIRMED_BY_KINEMATICS" if vlm_hypothesis == ball_dynamic else "REFINED_BY_RULES"

    return {
        'clip_id': clip_id,
        'actor_id': primary_actor_id,
        'vlm_hypothesis': vlm_hypothesis,
        'validated_activity': validated_activity,
        'decision_status': decision_status,
        'ball_displacement_px': round(ball_disp_px, 1),
        'ball_dx_px': round(dx, 1),
        'ball_dy_px': round(dy, 1),
        'player_dist_center_m': round(p_dist_center, 2),
        'player_displacement_m': round(p_disp_m, 2),
        'player_max_speed_kmh': round(p_max_speed, 2),
        'rules_applied': rules_triggered
    }
