# Dataset e Flussi Multimodali

Questa cartella raccoglie i campioni di dati pre-estratti e normalizzati utilizzati per la validazione e la replicabilita' degli esperimenti descritti nella tesi.

## Struttura della Cartella

- `ball_tracks/`: Coordinate bidimensionali del pallone per ciascun frame (formato: `frame x y`), estratte dal tracciamento ufficiale del match Tromso IL vs Tottenham Hotspurs (2013-11-28).
  - Include le clip `0000` - `0005` relative alla sequenza continua di avvio gara.
- `action_rois/`: 50 immagini ritagliate sull'Action Region of Interest con annotazioni visive semiotiche (Set-of-Marks) per il prompt visuale al Vision-Language Model:
  - Cerchio Verde: Posizione iniziale del calciatore attore (Kicker).
  - Polilinea Arancione: Traiettoria bidimensionale reale della palla.
  - Cerchio Rosso: Destinazione del passaggio o ricevitore potenziale.
- `sensors/`: Telemetria dei sensori indossabili ZXY a 20Hz:
  - `sensors_clip0000.csv`: Campionamenti della clip iniziale di calcio d'inizio.
  - `sensors_multi_clip.csv`: Telemetria sincronizzata per i primi 18 secondi di gioco (clip 0000 - 0005).
- `event_logs/`: Event Log strutturati conformemente agli standard di Process Mining (IEEE XES):
  - `event_log_multi_clip.csv`: Tracciamento multi-entita' (palla e calciatori attivi) per 18 secondi di gioco.
  - `event_log_full_first_half.csv`: Log globale di 6.451 eventi validati relativi all'intero primo tempo (767 clip, circa 38.4 minuti di partita).
  - `comparison_dataset_zero_shot_vs_domain_rules.csv`: Dataset comparativo sull'intero primo tempo (767 clip), con analisi puntuale della concordanza tra predizione visiva zero-shot e classificazione con regole di dominio.
  - `benchmark_sample_50_clips_comparison.csv`: Dataset di riferimento del benchmark su 50 eventi stratificati (dislocazione metrica palla, velocita' attore, etichetta di dominio, categoria di discrepanza).
  - `benchmark_50_clips_live_dual_qwen.csv`: Risultati della valutazione empirica duale eseguita realmente con la rete neurale Qwen2.5-VL-3B-Instruct (100 passaggi di inferenza: Pass A Zero-Shot vs Pass B In-Context Prompting con Knowledge Base).

## Riferimento Dataset Completo Alfheim
Il dataset originale completo include 767 clip video panoramiche (4450x2000 px) e oltre 493.000 campionamenti di sensori:
- Pettersen, S. A., et al. (2014). *Alfheim: An open multi-sensor dataset from professional soccer matches*. ACM MMSys.
- Repository di ateneo e server di calcolo: consultare la documentazione di laboratorio UNICAM.
