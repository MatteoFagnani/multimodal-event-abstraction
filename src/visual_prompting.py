"""
Modulo per l'estrazione e l'annotazione visiva dell'Action ROI
tramite tecnica Set-of-Marks (SoM) per il Visual Prompting del VLM.
"""

import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Optional

def extract_action_roi(
    image: np.ndarray,
    ball_df: pd.DataFrame,
    padding: int = 120,
    min_size: int = 400
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Estrae il bounding box ritagliato attorno alla traiettoria completa della palla.
    Ritorna il crop e le coordinate assolute (min_x, min_y, max_x, max_y).
    """
    b_min_x = int(ball_df['x'].min())
    b_max_x = int(ball_df['x'].max())
    b_min_y = int(ball_df['y'].min())
    b_max_y = int(ball_df['y'].max())
    
    # Calcolo centroide
    c_x = (b_min_x + b_max_x) // 2
    c_y = (b_min_y + b_max_y) // 2
    
    span_x = max(b_max_x - b_min_x + padding * 2, min_size)
    span_y = max(b_max_y - b_min_y + padding * 2, min_size)
    
    h_img, w_img = image.shape[:2]
    
    x1 = max(0, c_x - span_x // 2)
    x2 = min(w_img, c_x + span_x // 2)
    y1 = max(0, c_y - span_y // 2)
    y2 = min(h_img, c_y + span_y // 2)
    
    crop = image[y1:y2, x1:x2].copy()
    return crop, (x1, y1, x2, y2)

def annotate_set_of_marks(
    roi: np.ndarray,
    ball_df: pd.DataFrame,
    roi_bbox: Tuple[int, int, int, int],
    kicker_label: str = "P5",
    target_label: Optional[str] = "P12"
) -> np.ndarray:
    """
    Applica marcatori grafici semiotici (Set-of-Marks) sulla ROI:
    - Cerchio verde: Punto di partenza / Kicker
    - Polilinea arancione: Traiettoria continua della palla
    - Cerchio rosso: Zona di destinazione del passaggio
    """
    annotated = roi.copy()
    x1, y1, _, _ = roi_bbox
    
    # 1. Punti della traiettoria
    pts = []
    for _, row in ball_df.iterrows():
        px = int(row['x'] - x1)
        py = int(row['y'] - y1)
        pts.append((px, py))
        
    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [pts_arr], isClosed=False, color=(0, 140, 255), thickness=3)
    
    # 2. Marcatore di partenza (Verde)
    p_start = pts[0]
    cv2.circle(annotated, p_start, 18, (0, 255, 0), 3)
    cv2.putText(annotated, kicker_label, (p_start[0] - 12, p_start[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
    # 3. Marcatore di arrivo (Rosso)
    p_end = pts[-1]
    cv2.circle(annotated, p_end, 22, (0, 0, 255), 3)
    if target_label:
        cv2.putText(annotated, target_label, (p_end[0] - 12, p_end[1] + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
    return annotated
