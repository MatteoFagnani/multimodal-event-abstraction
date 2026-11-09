"""
Modulo di configurazione e costanti fisiche/geometriche del campo da gioco.
"""

from typing import Tuple

# Dimensioni del terreno di gioco regolamentare Alfheim Stadion (in metri)
PITCH_LENGTH_M: float = 105.0
PITCH_WIDTH_M: float = 68.0

# Punto di battuta del calcio d'inizio (coordinate metriche ZXY)
CENTER_SPOT_M: Tuple[float, float] = (52.5, 34.0)

# Risoluzione video panorama stitching
VIDEO_WIDTH_PX: int = 4450
VIDEO_HEIGHT_PX: int = 2000

# Soglie geometriche e cinematiche per il motore di Domain Knowledge
BALL_STATIONARY_THRESH_PX: float = 60.0       # Sotto questa soglia la palla e' ferma o in contesa locale
BALL_SHORT_PASS_THRESH_PX: float = 250.0      # Oltre questa soglia il movimento e' un passaggio o rilancio
BALL_LONG_PASS_THRESH_PX: float = 800.0       # Rilancio lungo / apertura di campo

PLAYER_CONTACT_RADIUS_M: float = 2.5          # Raggio massimo per considerare il giocatore a contatto con la palla
CENTER_CIRCLE_RADIUS_M: float = 9.15          # Raggio regolamentare del cerchio di centrocampo

# Frequenza di campionamento dei sensori indossabili ZXY
SENSOR_SAMPLING_RATE_HZ: int = 20
