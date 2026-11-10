# Event Abstraction con Domain Knowledge per il Process Mining su Flussi Multimodali nel Calcio Professionistico

## 1. Abstract e Motivazione Scientifica

L'applicazione del Process Mining in contesti sportivi complessi, quale il calcio professionistico, e' tradizionalmente ostacolata dal divario di astrazione (*abstraction gap*) tra i dati sensoriali a basso livello (telemetria corporea, tracciamento posizionale, flussi video continui) e gli eventi discreti ad alto livello semantico necessari per l'analisi dei processi (passaggi, pressing, riposizionamenti tattici, conclusioni a rete).

Questo progetto propone un'architettura ibrida e neuro-simbolica di **Event Abstraction** che integra:
1. **Flussi Multimodali Sincronizzati:** Video panoramici rettificati ad altissima definizione (4450x2000 px), tracciamento bidimensionale del pallone ad alta precisione e telemetria dei sensori indossabili ZXY a 20Hz (posizione $x,y$, velocita', accelerazione).
2. **Visual Prompting via Set-of-Marks (SoM):** Generazione automatica di ritagli centrate sull'azione (*Action ROI*) arricchite con annotatori grafici semiotici che guidano l'attenzione del modello visivo sulla dinamica di gioco.
3. **Classificazione Multimodale (Test A):** Inferenza zero-shot con Vision-Language Model di ultima generazione (*Qwen2.5-VL-3B-Instruct*).
4. **Validazione tramite Domain Knowledge Rules (Test B):** Motore deterministico basato su vincoli fisici, cinematici e regolamentari (IFAB) in grado di verificare le predizioni, eliminare le allucinazioni visive e classificare formalmente l'evento.
5. **Process Mining Globale:** Generazione di Event Log conformi allo standard IEEE XES e scoperta automatica di modelli di processo (*Directly-Follows Graphs*, reti di Petri, Social Network Analysis dei passaggi) tramite la libreria scientifica **PM4Py**.

La validazione e' stata condotta sul dataset accademico **Alfheim** (*Tromso IL vs Tottenham Hotspurs*, 2013-11-28), estendendo l'analisi dalla singola clip fino all'intero primo tempo di gioco (767 clip consecutive, circa 38.4 minuti, 6.451 eventi validati).

---

## 2. Architettura della Pipeline

La pipeline e' strutturata in cinque stadi sequenziali:

```
[Flussi Dati Alfheim]
   ├── Video Panorama (4450x2000)
   ├── Tracciamento Palla (frame-by-frame)
   └── Sensori ZXY 20Hz (x, y, v, a)
             │
             ▼
[1. Action ROI & Visual Prompting]
   ├── Calcolo centroide e dispersione della traiettoria
   └── Annotazione Set-of-Marks (Kicker Verde, Vettore Arancione, Target Rosso)
             │
             ▼
[2. Valutazione Ibrida Neuro-Simbolica]
   ├── Ramo A (VLM Zero-Shot): Predizione semantica visiva
   └── Ramo B (Domain Knowledge): Verifica cinematica (delta palla, velocita' attore, vincoli regolamentari)
             │
             ▼
[3. Generazione Event Log (IEEE XES)]
   ├── Traccia di possesso palla (Case: 'Possession_Phase')
   └── Tracce cinematiche individuali dei calciatori (Case: 'Player_ID_Trace')
             │
             ▼
[4. Process Mining & Analytics (PM4Py)]
   ├── Directly-Follows Graphs (DFG)
   ├── Social Network Analysis (Handover of Work / Rete dei Passaggi)
   └── Metriche atletiche aggregate (km percorsi, sprint ad alta velocita')
```

---

## 3. Struttura della Repository

```
thesis-multimodal-event-abstraction/
├── .gitignore                      # Configurazione esclusioni Git
├── README.md                       # Documentazione scientifica del progetto
├── requirements.txt                # Dipendenze software Python
├── run_experiments.py              # Script CLI unificato per la replicabilita' degli esperimenti
├── notebooks/
│   └── pipeline_experimentation.ipynb  # Notebook interattivo per replicare i test (CPU o GPU)
├── src/
│   ├── __init__.py                 # Inizializzazione package
│   ├── config.py                   # Parametri metrici del campo e costanti di soglia
│   ├── data_loader.py              # Ingestione e filtraggio dati telemetrici e tracciamento
│   ├── visual_prompting.py         # Algoritmi di ritaglio ROI e annotazione grafica Set-of-Marks
│   ├── domain_rules.py             # Motore a regole neuro-simboliche di dominio calcistico
│   ├── multi_clip_engine.py        # Elaborazione sequenziale su multi-clip (0000 - 0005)
│   ├── full_match_engine.py        # Ingestione ed Event Abstraction su scala intero match (767 clip)
│   └── process_mining_analysis.py  # Integrazione PM4Py, scoperta DFG e grafo dei passaggi (SNA)
├── data/
│   ├── README.md                   # Documentazione sui flussi dati
│   ├── ball_tracks/                # Coordinate di tracciamento palla (clip 0000 - 0005)
│   ├── action_rois/                # Immagini ROI con annotatori Set-of-Marks
│   ├── sensors/                    # Telemetria sensori ZXY a 20Hz (clip 0000 e multi-clip)
│   └── event_logs/                 # Event Log strutturati (multi-clip e 6.451 eventi full match)
└── results/
    ├── match_statistics.json       # Statistiche descrittive e prestazionali calcolate
    ├── process_map_dfg.png         # Modello di processo Directly-Follows Graph
    ├── pass_network_full_match.png # Grafo orientato della rete dei passaggi (Handover of Work)
    └── activity_distribution_full_match.png # Distribuzione percentuale delle attivita'
```

---

## 4. Requisiti di Sistema e Installazione

Il codice e' compatibile con ambienti Linux, macOS e Windows. Si consiglia l'uso di Python 3.10 o versioni successive.

### Creazione dell'ambiente virtuale

```bash
# Clonazione della repository
git clone <URL_REPOSITORY>
cd thesis-multimodal-event-abstraction

# Creazione ambiente virtuale isolato
python -m venv venv

# Attivazione ambiente
# Su Linux/macOS:
source venv/bin/activate
# Su Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Installazione delle dipendenze
pip install -r requirements.txt
```

---

## 5. Replicabilita' degli Esperimenti

L'intera sperimentazione puo' essere eseguita sia tramite riga di comando sia tramite Jupyter Notebook.

### Esecuzione da Riga di Comando (CLI)

Il modulo `run_experiments.py` consente di eseguire singole fasi sperimentali o la pipeline completa:

```bash
# Esecuzione completa di tutti gli stadi sperimentali
python run_experiments.py --stage all

# Esecuzione mirata di un singolo stadio:
python run_experiments.py --stage 1    # Validazione Clip 0000 (Test A vs Test B)
python run_experiments.py --stage 2    # Estrazione Multi-Clip (0000-0005, 18 secondi di gioco)
python run_experiments.py --stage 3    # Process Mining su scala globale (6.451 eventi)
```

### Esecuzione via Jupyter Notebook / Google Colab

Il file `notebooks/pipeline_experimentation.ipynb` contiene l'intera trattazione guidata:
- Supporta l'esecuzione **sia su CPU sia su GPU**.
- Se eseguito su macchine prive di acceleratore grafico (o in caso di quota GPU esaurita su Google Colab), il notebook attiva automaticamente la modalita' *Dual-Mode*, impiegando il responso visivo certificato del modello VLM per il Test A ed eseguendo in tempo reale su CPU tutte le regole di dominio, l'Event Abstraction e l'analisi con PM4Py in meno di 5 secondi.

---

## 6. Risultati Sperimentali e Discussione

### Stadio 1: Validazione su Clip Singola (Clip 0000 - Calcio d'Inizio)
- **Baseline VLM (Test A):** Il modello zero-shot riconosce la macro-azione di *Kickoff_and_Pass* grazie al visual prompting semiotico.
- **Validazione Neuro-Simbolica (Test B):** Le regole di dominio verificano la posizione dell'attore primario ($d = 4.49\text{ m}$ dal dischetto di centrocampo, all'interno del cerchio regolamentare di raggio $9.15\text{ m}$) e lo spostamento netto della palla ($\Delta = 459.7\text{ px}$). L'evento viene formalmente confermato (`CONFIRMED_BY_KINEMATICS`).

### Stadio 2: Sequenza Multi-Clip (Clip 0000 - 0005, 18 Secondi)
La pipeline ricostruisce la catena causale completa del possesso palla iniziale:
$$\text{Kickoff\_and\_Pass (P5)} \longrightarrow \text{Ball\_Control\_and\_Carry (P12)} \longrightarrow \text{Ball\_Control\_and\_Carry (P11)} \longrightarrow \text{Ball\_Control\_and\_Carry (P2)} \longrightarrow \text{Long\_Forward\_Pass (P4)} \longrightarrow \text{Backward\_Pass (P14)}$$
La telemetria a 20Hz documenta contestualmente lo scatto ad alta intensita' del ricevitore primario ($v_{\max} = 27.2\text{ km/h}$).

### Stadio 3: Process Mining su Scala Globale (Intero Primo Tempo)
- **Campione di analisi:** 767 clip consecutive (57.525 frame video, 493.663 campionamenti ZXY, 38.35 minuti di gioco effettivo).
- **Event Log estratto:** 6.451 eventi validati (767 eventi palla, 5.684 eventi individuali dei calciatori).
- **Compressione ed efficienza dati:** I dati grezzi originali ammontano a circa 6.5 GB di video e oltre 65 MB di telemetria tabellare. L'Event Log finale sintetizza l'intera dinamica di gioco in un file da circa 824 KB (oltre il **99.98% di riduzione del volume dati**), mantenendo intatta la fedelta' semantica del match.
- **Distribuzione delle attivita' con la palla:**
  - *Ball_Control_and_Carry:* 239 occorrenze (31.2%)
  - *Stationary_Play:* 158 occorrenze (20.6%)
  - *Forward_Pass:* 152 occorrenze (19.8%)
  - *Backward_Pass:* 131 occorrenze (17.1%)
  - *Long_Forward_Pass / Clearance:* 78 occorrenze (10.2%)
  - *Lateral_Pass / Kickoff:* 9 occorrenze (1.1%)

---

## 7. Riferimenti Bibliografici

1. **van der Aalst, W. M. P.** (2016). *Process Mining: Data Science in Action*. Springer.
2. **Pettersen, S. A., Johansen, D., Johansen, H., Berg-Johansen, V., Gaddam, V. R., Mortensen, A., Langseth, R., Griwodz, C., Stensland, H. K., & Halvorsen, P.** (2014). *Alfheim: An open multi-sensor dataset from professional soccer matches*. In Proceedings of the 5th ACM Multimedia Systems Conference (MMSys '14), pp. 200–205.
3. **Ferrara, A., et al.** (2024). *AA-STAL: Spatial-Temporal Action Localization with Visual Prompting and Set-of-Marks*. Computer Vision and Pattern Recognition.
4. **Qwen Team** (2025). *Qwen2.5-VL: Enhancing Vision-Language Capabilities for Dynamic and High-Resolution Multimodal Understanding*. arXiv preprint.
5. **Berti, A., et al.** (2019). *Process Mining for Python (PM4Py): An open-source library for process analysis*. CEUR Workshop Proceedings.
