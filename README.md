# Event Abstraction e Process Mining su Flussi Multimodali nel Calcio

Repository di test e sperimentazione per la pipeline di **Event Abstraction** e **Process Mining** applicata a flussi multimodali nel calcio professionistico, basata sul dataset accademico **Alfheim** (*Tromso IL vs Tottenham Hotspurs*, 28-11-2013).

La repository contiene il codice sorgente, i campioni di dati pre-estratti, i notebook per l'esecuzione su Google Colab e gli artefatti prodotti dai test sperimentali (Event Log conformi a IEEE XES, modelli di processo scoperti con PM4Py, metriche e tabelle comparative).

---

## Struttura della Repository

```text
thesis-multimodal-event-abstraction/
├── README.md                           # Documentazione tecnica della repository
├── requirements.txt                    # Dipendenze software Python
├── run_experiments.py                  # Script CLI per l'esecuzione modulare dei test
├── notebooks/
│   └── pipeline_experimentation.ipynb  # Notebook interattivo per la riproduzione su Google Colab
├── src/
│   ├── __init__.py                     # Inizializzazione package Python
│   ├── config.py                       # Costanti metriche, coordinate del campo e soglie
│   ├── data_loader.py                  # Caricamento e filtraggio di sensori ZXY e tracciamento palla
│   ├── visual_prompting.py             # Algoritmi di ritaglio Action ROI e annotazioni Set-of-Marks
│   ├── domain_rules.py                 # Motore a regole di dominio (cinematica e regolamento IFAB)
│   ├── multi_clip_engine.py            # Elaborazione della sequenza continua (clip 0000 - 0005)
│   ├── full_match_engine.py            # Elaborazione sull'intero primo tempo (767 clip)
│   ├── process_mining_analysis.py      # Scoperta di grafi DFG e Social Network Analysis con PM4Py
│   ├── benchmark_comparison.py         # Test comparativo su scala intero match (767 clip)
│   └── benchmark_50_clips.py           # Benchmark su 50 eventi e driver di inferenza neurale
├── data/
│   ├── README.md                       # Dettaglio sui dati inclusi nella cartella
│   ├── action_rois/                    # 50 immagini ritagliate con annotatori Set-of-Marks
│   ├── ball_tracks/                    # Coordinate frame-by-frame della palla (clip 0000 - 0005)
│   ├── sensors/                        # Estratti di telemetria dei sensori ZXY a 20Hz
│   └── event_logs/
│       ├── event_log_multi_clip.csv    # Event Log per i primi 18 secondi di gioco
│       ├── event_log_full_first_half.csv # Event Log globale di 6.451 eventi (767 clip)
│       ├── comparison_dataset_zero_shot_vs_domain_rules.csv # Dataset comparativo (767 clip)
│       ├── benchmark_sample_50_clips_comparison.csv # Dataset di riferimento per il test su 50 clip
│       └── benchmark_50_clips_live_dual_qwen.csv    # Risultati dell'inferenza neurale live su Qwen
└── results/
    ├── match_statistics.json           # Statistiche descrittive e prestazionali del match
    ├── process_map_dfg.png             # Directly-Follows Graph scoperto con PM4Py
    ├── pass_network_full_match.png     # Grafo orientato della rete dei passaggi tra calciatori
    ├── activity_distribution_full_match.png # Distribuzione percentuale delle attivita'
    ├── comparison_summary_metrics.json # Metriche di concordanza globale (767 clip)
    ├── confusion_matrix_comparison.csv # Matrice di contingenza empirica (767 clip)
    ├── latex_table_comparison.tex      # Tabella LaTeX della tassonomia discrepanze
    ├── benchmark_sample_50_metrics.json # Metriche del benchmark su 50 clip
    ├── latex_table_50_clips_benchmark.tex # Tabella LaTeX comparativa dei 50 eventi
    ├── benchmark_50_clips_live_dual_metrics.json # Metriche dell'esperimento neurale live con Qwen
    └── latex_table_50_clips_live_dual_qwen.tex   # Tabella LaTeX con il confronto puntuale live
```

---

## Contenuto dei File e delle Cartelle

### 1. Codice Sorgente (`src/` e Root)
- `run_experiments.py`: Interfaccia a riga di comando per riprodurre le varie fasi della pipeline.
- `src/config.py`: Definisce le dimensioni standard del campo da gioco (105m x 68m), la risoluzione video panoramica (4450x2000 px) e le soglie di velocità e spostamento per la classificazione delle azioni.
- `src/data_loader.py`: Funzioni di parsing per i file di tracciamento palla e per la telemetria ZXY a 20Hz (posizione x/y, velocità in km/h, accelerazione, identificativo giocatore).
- `src/visual_prompting.py`: Genera i ritagli *Action ROI* intorno alla traiettoria del pallone e sovrappone gli indicatori *Set-of-Marks* (cerchio verde per il giocatore partente, polilinea arancione per la traiettoria, cerchio rosso per l'area di arrivo).
- `src/domain_rules.py`: Insieme di funzioni che applicano vincoli cinematici e regole del gioco del calcio (IFAB) per validare o raffinare le ipotesi di azione (es. verifica della posizione rispetto al centrocampo, spostamento minimo/massimo della palla).
- `src/multi_clip_engine.py`: Costruisce la traccia continua di eventi per le prime 6 clip della partita (18 secondi di gioco effettivo).
- `src/full_match_engine.py`: Gestisce l'estrazione degli eventi per l'intero primo tempo di gara (767 clip), generando l'Event Log per il Process Mining e calcolando statistiche atletiche individuali.
- `src/process_mining_analysis.py`: Interfaccia con la libreria scientifica PM4Py per generare grafi Directly-Follows Graph (DFG) e calcolare la Social Network Analysis (Handover of Work) sui passaggi tra compagni di squadra.
- `src/benchmark_comparison.py`: Modulo per il confronto sistematico tra l'approccio puramente visivo (Zero-Shot) e quello con regole di dominio su tutte le 767 clip del match.
- `src/benchmark_50_clips.py`: Gestisce il benchmark stratificato su 50 eventi rappresentativi e l'interrogazione reale del modello neurale *Qwen2.5-VL*.

### 2. Dati (`data/`)
- `action_rois/`: 50 immagini ritagliate in formato JPG con visual prompting applicato, relative a 50 momenti di gioco rappresentativi (calcio d'inizio, passaggi in avanti, retropassaggi, controlli palla, palle ferme, lanci lunghi, rinvii difensivi).
- `ball_tracks/`: Tracciamento palla bidimensionale estratto frame per frame per le clip 0000, 0001, 0002, 0003, 0004, 0005.
- `sensors/`: Campionamenti dei sensori ZXY a 20Hz sincronizzati per la clip 0000 e per i primi 18 secondi di gioco (clip 0000 - 0005).
- `event_logs/`:
  - `event_log_multi_clip.csv`: Event Log della sequenza di avvio gara (71 record multi-entità).
  - `event_log_full_first_half.csv`: Event Log completo del primo tempo di gara contenente 6.451 eventi validati (767 eventi palla e 5.684 eventi individuali dei calciatori).
  - `comparison_dataset_zero_shot_vs_domain_rules.csv`: Dataset di comparazione su tutte le 767 clip (predizione visiva vs classificazione con regole di dominio, tassonomia della discrepanza e impatto sul Process Mining).
  - `benchmark_sample_50_clips_comparison.csv`: Dataset di confronto sui 50 eventi selezionati.
  - `benchmark_50_clips_live_dual_qwen.csv`: Risultati delle 100 inferenze eseguite realmente dal modello Qwen2.5-VL (Pass A solo visivo vs Pass B con dati sensoriali e regole).

### 3. Risultati e Modelli Generati (`results/`)
- `process_map_dfg.png`: Modello di processo Directly-Follows Graph ricavato dagli eventi palla del primo tempo.
- `pass_network_full_match.png`: Grafo orientato delle interazioni e transizioni di passaggio tra i calciatori in campo.
- `activity_distribution_full_match.png`: Grafico a barre con la frequenza percentuale delle macro-attività estratte.
- `match_statistics.json`: File JSON riassuntivo contenente indicatori prestazionali (distanze percorse in km, velocità massime, numero di sprint) e frequenze di transizione.
- File di metriche JSON (`comparison_summary_metrics.json`, `benchmark_sample_50_metrics.json`, `benchmark_50_clips_live_dual_metrics.json`).
- Tabelle in formato LaTeX (`latex_table_comparison.tex`, `latex_table_50_clips_benchmark.tex`, `latex_table_50_clips_live_dual_qwen.tex`) pronte per essere incluse in documenti accademici.

---

## Installazione e Requisiti

Il codice richiede Python 3.10 o versioni successive.

```bash
# 1. Clonazione della repository
git clone https://github.com/<username>/thesis-multimodal-event-abstraction.git
cd thesis-multimodal-event-abstraction

# 2. Creazione dell'ambiente virtuale
python -m venv venv

# Attivazione su Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Oppure su Linux/macOS:
source venv/bin/activate

# 3. Installazione delle librerie
pip install -r requirements.txt
```

---

## Esecuzione dei Test

### 1. Da Riga di Comando (`run_experiments.py`)

Lo script consente di eseguire i test per singoli stadi o nell'insieme:

```bash
# Esecuzione completa di tutti i test (Stadi 1, 2, 3, 4, 5):
python run_experiments.py --stage all

# Esecuzione di uno specifico stadio:
python run_experiments.py --stage 1    # Singola clip (Clip 0000 - Calcio d'inizio)
python run_experiments.py --stage 2    # Sequenza multi-clip (0000 - 0005, 18 secondi)
python run_experiments.py --stage 3    # Process Mining su scala globale (6.451 eventi full match)
python run_experiments.py --stage 4    # Dataset comparativo globale su 767 clip
python run_experiments.py --stage 5    # Benchmark sperimentale su 50 clip
```

Se si dispone di una GPU locale e delle librerie `transformers` e `torch`, e' possibile eseguire l'inferenza reale del modello Qwen sulle Action ROI:
```bash
python run_experiments.py --stage 5 --live-vlm
```

### 2. Tramite Google Colab (`notebooks/pipeline_experimentation.ipynb`)

Il notebook `pipeline_experimentation.ipynb` permette di eseguire l'intera pipeline in ambiente interattivo:
- **Modalità CPU:** Se non è disponibile una GPU, esegue istantaneamente tutti i test, l'applicazione delle regole di dominio, la scoperta di processo con PM4Py e l'analisi dei benchmark utilizzando i responsi certificati del modello.
- **Modalità GPU (T4):** Permette di caricare realmente il modello multimodale `Qwen/Qwen2.5-VL-3B-Instruct` e svolgere le inferenze visive e multimodali dal vivo.
- Include una cella finale di esportazione che raggruppa tutti i risultati in un unico archivio `.zip` scaricabile.

---

## Descrizione dei Test Inclusi

1. **Stage 1 - Test su Singola Clip (Clip 0000):** Verifica della prima azione di gioco (calcio d'inizio), confrontando la predizione visiva con la verifica di posizione dell'attore rispetto al centro del campo e lo spostamento della palla.
2. **Stage 2 - Sequenza Continua Multi-Clip (Clip 0000 - 0005):** Elaborazione sequenziale dei primi 18 secondi di gioco per verificare la concatenazione temporale delle attività di possesso palla e dei dati fisici dei calciatori.
3. **Stage 3 - Process Mining su Scala Globale (Intero Primo Tempo):** Elaborazione delle 767 clip del primo tempo di gara (circa 38.4 minuti effettivi), generazione dell'Event Log di 6.451 eventi e scoperta del modello di processo Directly-Follows Graph tramite PM4Py.
4. **Stage 4 - Benchmark Globale su 767 Clip:** Confronto tra classificazione puramente visiva e classificazione mediata da regole di dominio, con calcolo del tasso di concordanza globale e censimento delle discrepanze (inversioni direzionali, allucinazioni di movimento su palla ferma, ecc.).
5. **Stage 5 - Benchmark su 50 Eventi e Test Neurale Live (Qwen2.5-VL):** Test di classificazione condotto su un campione stratificato di 50 clip rappresentative in due modalità:
   - *Passaggio A (Zero-Shot Visivo):* Solo immagine Action ROI con annotazione grafica.
   - *Passaggio B (Multimodale con Regole):* Immagine arricchita con telemetria numerica a 20Hz e vincoli di dominio.
