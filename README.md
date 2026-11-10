# Event Abstraction con Domain Knowledge per il Process Mining su Flussi Multimodali nel Calcio Professionistico

**Tesi di Laurea Triennale in Informatica | Universita' di Camerino (UNICAM)**  
**Candidato:** Matteo Fagnani  
**Relatrice:** Prof.ssa Barbara Re  
**Dataset di Riferimento:** Alfheim Dataset (Tromso IL vs Tottenham Hotspurs, 2013-11-28)

---

## 1. Abstract e Obiettivi Scientifici

L'applicazione del Process Mining all'analisi del calcio professionistico presenta una criticita' metodologica fondamentale nota come *abstraction gap*: il disallineamento ontologico tra dati sensoriali a basso livello (coordinate cartesiane frame-per-frame, telemetria corporea ad alta frequenza, registrazioni video continue) e gli eventi discreti ad alto livello semantico richiesti dagli algoritmi di process discovery (passaggi, controlli orientati, fasi di possesso, transizioni tattiche).

Questo progetto sviluppa e valida un framework di **Event Abstraction Ibrido e Neuro-Simbolico** progettato per:
1. **Integrare ed Allineare Flussi Multimodali Eterogenei:** Flussi video panoramici rettificati ad alta definizione (4450x2000 pixel), coordinate di tracciamento continuo del pallone (campionate a 25Hz) e telemetria metrica dei sensori indossabili ZXY a 20Hz (posizione $x,y$, velocita' scalare, accelerazione e dislocazione).
2. **Guidare l'Attenzione Visiva tramite Visual Prompting:** Generazione deterministica di ritagli spaziali centrate sull'azione (*Action Region of Interest* - Action ROI), arricchite con evidenziatori semiotici *Set-of-Marks* (SoM): cerchio verde sul calciatore attore (*kicker*), vettore policromatico arancione della traiettoria reale e cerchio rosso sulla zona di destinazione.
3. **Valutare Empiricamente Modelli di Vision-Language (VLM):** Sottoporre a test formale il modello multimodale di ultima generazione *Qwen2.5-VL-3B-Instruct*, quantificandone i limiti percettivi (allucinazioni di movimento su palla ferma, inversioni direzionali prospettiche, ambiguita' tra controllo e passaggio).
4. **Implementare un Motore di Validazione Neuro-Simbolico con Domain Knowledge:** Formalizzare in un insieme rigoroso di regole di dominio cinematiche e regolamentari (regolamento ufficiale IFAB) la conoscenza a priori del gioco del calcio, correggendo le allucinazioni del VLM e garantendo la verita' a terra dell'evento.
5. **Generare Event Log Standard IEEE XES ed Eseguire Process Mining su Scala Globale:** Trasformare l'intero primo tempo di gioco (767 clip, 38.35 minuti effettivi) in un Event Log compresso e strutturato, scoprendo tramite la libreria scientifica **PM4Py** il modello di processo formale (*Directly-Follows Graph* - DFG) e la rete sociale delle interazioni di passaggio (*Social Network Analysis*).

---

## 2. Architettura del Sistema

Il flusso di elaborazione e' articolato in stadi cooperanti:

```
[Flussi Dati Multimodali Alfheim]
   ├── Video Panoramico HD (4450x2000 px)
   ├── Tracciamento Palla 2D (frame-by-frame)
   └── Sensori Indossabili ZXY 20Hz (x, y, v, a)
             │
             ▼
[1. Action ROI & Visual Prompting (Set-of-Marks)]
   ├── Calcolo del centroide di movimento della palla
   └── Annotazione semiotica (Kicker Verde, Vettore Arancione, Target Rosso)
             │
             ▼
[2. Valutazione Ibrida Neuro-Simbolica]
   ├── Ramo A (VLM Puro Zero-Shot): Predizione semantica visuale
   └── Ramo B (Domain Knowledge): Verifica cinematica e regolamentare IFAB
             │
             ▼
[3. Event Abstraction & Generazione Log IEEE XES]
   ├── Traccia di possesso collettivo (Case: 'Possession_Phase')
   └── Tracce prestazionali individuali (Case: 'Player_ID_Trace')
             │
             ▼
[4. Process Mining & Analisi Tattica (PM4Py)]
   ├── Directly-Follows Graph (DFG) con frequenze di transizione
   ├── Social Network Analysis (Handover of Work / Rete dei Passaggi)
   └── Indicatori fisici e atletici (distanze, sprint ad alta velocita')
```

---

## 3. Struttura del Repository

La cartella del progetto e' strutturata in modo modulare, auto-consistente e priva di file ridondanti:

```
thesis-multimodal-event-abstraction/
├── .gitignore                          # Esclusioni per bytecode, cache e file pesanti
├── README.md                           # Relazione scientifica ed architetturale del progetto
├── requirements.txt                    # Elenco delle dipendenze Python replicabili
├── run_experiments.py                  # Script CLI principale per l'esecuzione degli esperimenti
├── notebooks/
│   └── pipeline_experimentation.ipynb  # Notebook sperimentale (eseguibile su Google Colab o locale)
├── src/
│   ├── __init__.py                     # Package init
│   ├── config.py                       # Costanti metriche, dimensioni del campo e soglie cinematiche
│   ├── data_loader.py                  # Ingestione e filtraggio dei campionamenti telemetrici
│   ├── visual_prompting.py             # Algoritmi di ritaglio Action ROI e annotazione Set-of-Marks
│   ├── domain_rules.py                 # Motore inferenziale a regole di dominio (IFAB + cinematica)
│   ├── multi_clip_engine.py            # Motore di elaborazione sequenziale multi-clip (0000 - 0005)
│   ├── full_match_engine.py            # Motore di astrazione su scala globale (intero primo tempo, 767 clip)
│   ├── process_mining_analysis.py      # Scoperta di processi con PM4Py, grafi DFG e reti sociali
│   ├── benchmark_comparison.py         # Modulo di benchmarking globale (767 clip)
│   └── benchmark_50_clips.py           # Benchmark stratificato su 50 eventi e driver di inferenza duale
├── data/
│   ├── README.md                       # Documentazione descrittiva sui dati
│   ├── ball_tracks/                    # Tracciamenti bidimensionali palla (clip 0000 - 0005)
│   ├── action_rois/                    # 50 Action ROI annotate con visual prompting Set-of-Marks
│   ├── sensors/                        # Slices di telemetria a 20Hz (clip 0000 e multi-clip 18s)
│   └── event_logs/
│       ├── event_log_multi_clip.csv    # Event Log multi-entita' per i primi 18 secondi di gioco
│       ├── event_log_full_first_half.csv # Event Log globale di 6.451 eventi validati (767 clip)
│       ├── comparison_dataset_zero_shot_vs_domain_rules.csv # Dataset comparativo full-match (767 clip)
│       ├── benchmark_sample_50_clips_comparison.csv # Dataset di riferimento benchmark (50 eventi)
│       └── benchmark_50_clips_live_dual_qwen.csv    # Valutazione empirica duale live con Qwen2.5-VL
└── results/
    ├── match_statistics.json           # Statistiche fisiche e tattiche estratte dall'intero match
    ├── process_map_dfg.png             # Modello di processo Directly-Follows Graph scoperto
    ├── pass_network_full_match.png     # Grafo orientato della rete dei passaggi tra calciatori
    ├── activity_distribution_full_match.png # Istogramma della distribuzione delle attivita' astratte
    ├── comparison_summary_metrics.json # Metriche quantitative di concordanza globale (767 clip)
    ├── confusion_matrix_comparison.csv # Matrice di contingenza tra approccio visivo e regole
    ├── latex_table_comparison.tex      # Tabella LaTeX della tassonomia errori su 767 clip
    ├── benchmark_sample_50_metrics.json # Metriche aggregate del benchmark sui 50 eventi
    ├── latex_table_50_clips_benchmark.tex # Tabella LaTeX comparativa dei 50 eventi stratificati
    ├── benchmark_50_clips_live_dual_metrics.json # Metriche dell'esperimento neurale live con Qwen
    └── latex_table_50_clips_live_dual_qwen.tex   # Tabella LaTeX dell'esperimento neurale live
```

---

## 4. Installazione e Replicabilita'

Il codice e' verificato su Python 3.10 e 3.11 su ambienti Windows, Linux e Google Colab.

### Installazione Locale

```bash
# 1. Clonazione del repository
git clone https://github.com/<username>/thesis-multimodal-event-abstraction.git
cd thesis-multimodal-event-abstraction

# 2. Creazione dell'ambiente virtuale
python -m venv venv

# Attivazione ambiente su Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Oppure su Linux/macOS:
source venv/bin/activate

# 3. Installazione delle dipendenze
pip install -r requirements.txt
```

### Esecuzione Tramite Interfaccia a Riga di Comando (CLI)

Il comando `run_experiments.py` consente di riprodurre ciascuna fase sperimentale:

```bash
# Esecuzione completa di tutti gli stadi (1, 2, 3, 4, 5):
python run_experiments.py --stage all

# Esecuzione mirata di un singolo stadio:
python run_experiments.py --stage 1    # Stadio 1: Validazione singola clip (Kickoff, clip 0000)
python run_experiments.py --stage 2    # Stadio 2: Sequenza multi-clip (0000 - 0005, 18 secondi)
python run_experiments.py --stage 3    # Stadio 3: Process Mining su 6.451 eventi (Full Match)
python run_experiments.py --stage 4    # Stadio 4: Benchmark globale e matrice comparativa (767 clip)
python run_experiments.py --stage 5    # Stadio 5: Benchmark sperimentale su 50 eventi stratificati
```

Per eseguire l'inferenza neurale *live* sul modello Qwen2.5-VL-3B-Instruct sfruttando la GPU locale (se presente):
```bash
python run_experiments.py --stage 5 --live-vlm
```

### Esecuzione Interattiva su Google Colab

Il notebook `notebooks/pipeline_experimentation.ipynb` e' concepito per la massima fruibilita' accademica:
- Include una modalita' **Dual-Mode**: se eseguito su CPU standard (es. quota gratuita GPU esaurita su Colab), completa l'intera pipeline di Process Mining, l'Event Abstraction e i benchmark in circa 3 secondi utilizzando i responsi certificati del modello.
- Se eseguito su GPU T4, istanzia realmente i pesi di `Qwen/Qwen2.5-VL-3B-Instruct` in precisione float16 / 4-bit ed esegue l'inferenza visiva e multimodale in tempo reale.

---

## 5. Sintesi dei Test Condotti e Risultati Sperimentali

La sperimentazione ha seguito una progressione metodologica rigorosa, articolata in cinque stadi scientifici:

### Stadio 1: Validazione su Clip Singola (Clip 0000 - Calcio d'Inizio)
- **Obiettivo:** Verificare se il solo prompt visivo sia sufficiente a catturare la natura regolamentare dell'avvio partita.
- **Esito VLM (Test A):** Qwen2.5-VL riconosce l'azione macroscopica come *Kickoff_and_Pass*.
- **Validazione Neuro-Simbolica (Test B):** Le regole di dominio verificano che la posizione iniziale del calciatore (Player 5) sia a $4.49\text{ m}$ dal dischetto di centrocampo (entro il cerchio di centrocampo di raggio regolamentare $9.15\text{ m}$) e che lo spostamento palla sia significativo ($\Delta = 459.7\text{ px}$). L'evento viene formalmente confermato (`CONFIRMED_BY_KINEMATICS`).

### Stadio 2: Sequenza Multi-Clip Continua (Clip 0000 - 0005, 18 Secondi di Gioco)
- **Obiettivo:** Dimostrare la capacita' di estrarre una catena causale ininterrotta di possesso palla per il Process Mining.
- **Risultato:** La pipeline ricostruisce la traiettoria tattica completa:
  $$\text{Kickoff\_and\_Pass (P5)} \longrightarrow \text{Ball\_Control\_and\_Carry (P12)} \longrightarrow \text{Ball\_Control\_and\_Carry (P11)} \longrightarrow \text{Ball\_Control\_and\_Carry (P2)} \longrightarrow \text{Long\_Forward\_Pass (P4)} \longrightarrow \text{Backward\_Pass (P14)}$$
- **Integrazione sensoriale:** La telemetria ZXY rileva contestualmente l'accelerazione del ricevitore (Player 4) che raggiunge la velocita' di picco di $27.2\text{ km/h}$.

### Stadio 3: Process Mining su Scala Globale (Intero Primo Tempo, 767 Clip)
- **Volume dati originario:** Oltre 6.5 GB di video panoramico non compresso e 493.663 campionamenti telemetrici (38.35 minuti di gioco).
- **Event Log sintetizzato:** Un file XES da 824 KB contenente **6.451 eventi validati** (767 eventi della palla, 5.684 eventi individuali dei calciatori attivi), con un abbattimento volumetrico superiore al **99.98%**.
- **Distribuzione delle attivita' della palla:**
  - *Ball_Control_and_Carry:* 239 eventi (31.2%)
  - *Stationary_Play:* 158 eventi (20.6%)
  - *Forward_Pass:* 152 eventi (19.8%)
  - *Backward_Pass:* 131 eventi (17.1%)
  - *Clearance_or_Long_Back_Pass:* 39 eventi (5.1%)
  - *Long_Forward_Pass:* 39 eventi (5.1%)
  - *Lateral_Pass:* 8 eventi (1.0%)
  - *Kickoff_and_Pass:* 1 evento (0.1%)
- **Process Discovery (PM4Py):** Generazione del Directly-Follows Graph (DFG) con 46 transizioni dirette identificate.
- **Social Network Analysis (Handover of Work):** Mappatura della rete di passaggio tra i calciatori, individuando Player 11 (116 sprint, $v_{\max} = 34.8\text{ km/h}$) e Player 12 ($v_{\max} = 38.3\text{ km/h}$) come i nodi centrali della manovra.

### Stadio 4: Benchmark Globale su 767 Clip (Zero-Shot vs Domain Rules)
Il confronto sistematico tra l'approccio puramente visivo (Zero-Shot VLM) e l'approccio ibrido neuro-simbolico evidenzia:
- **Tasso di Concordanza Globale (Agreement Rate):** $63.75\%$ (489 clip concordanti).
- **Tasso di Discrepanza Risolto:** $36.25\%$ (278 clip corrette dalle regole di dominio).
- **Tassonomia delle anomalie visive neutralizzate:**
  - *Allucinazione di Movimento su Gioco Stazionario (`STATIONARY_HALLUCINATION`):* 109 clip ($14.21\%$). Il VLM scambia la corsa del giocatore vicino alla palla per un passaggio, mentre la telemetria certifica $\Delta \text{ball} < 50\text{ px}$.
  - *Inversione Direzionale Prospettica (`DIRECTIONAL_INVERSION`):* 68 clip ($8.87\%$). Il VLM interpreta retropassaggi difensivi verso la propria porta come passaggi in avanti per via dell'orientamento corporeo prospettico dei compagni.
  - *Ambiguita' Controllo/Passaggio (`CONTROL_PASS_AMBIGUITY`):* 44 clip ($5.74\%$).
  - *Ambiguita' Passaggio/Controllo (`PASS_CONTROL_AMBIGUITY`):* 29 clip ($3.78\%$).
  - *Inversione Tattica Rinvio/Lancio (`TACTICAL_INVERSION`):* 19 clip ($2.48\%$).
  - *Ambiguita' Laterale (`LATERAL_AMBIGUITY`):* 8 clip ($1.04\%$).
  - *Omissione Regolamentare (`REGULATORY_OMISSION`):* 1 clip ($0.13\%$).
- **Impatto sul Process Mining:** Eliminazione di 109 cicli fittizi (*false loops*) nel modello DFG, rettifica di 87 transizioni invertite di flusso e corretta granularita' per 81 attivita'.

### Stadio 5: Benchmark Sperimentale su 50 Eventi e Valutazione Neurale Live con Qwen2.5-VL
Per convalidare scientificamente il comportamento del modello neurale, e' stato allestito un benchmark controllato su un campione stratificato di 50 eventi tattici critici (1 calcio d'inizio, 10 retropassaggi difensivi, 10 passaggi in avanti, 10 controlli orientati, 10 palle ferme con atleti in corsa a oltre 12 km/h, 5 lanci lunghi, 4 rinvii difensivi).

L'esperimento e' stato condotto realmente su Google Colab con accelerazione GPU NVIDIA Tesla T4, sottoponendo Qwen2.5-VL a **100 passaggi di inferenza** (2 inferenze indipendenti per ciascuna delle 50 clip):
1. **Passaggio A - Zero-Shot Visivo Puro (Senza Knowledge Base):**
   - Fornendo al modello la sola Action ROI ritagliata con Set-of-Marks, il modello neurale collassa su una predizione monocorde: predice `Forward_Pass` in **44 clip su 50 (88.0%)**.
   - *Spiegazione scientifica:* Il VLM e' affetto da un forte bias induttivo visivo. La presenza del vettore tracciato sulla ROI induce la rete ad allucinare un passaggio in avanti perfino su palle completamente ferme ($\Delta = 0.0\text{ px}$) o su retropassaggi verso la propria porta ($\Delta x = -506\text{ px}$).
2. **Passaggio B - In-Context Prompting con Knowledge Base e Telemetria 20Hz:**
   - Iniettando nel prompt i dati dei sensori (dislocazione della palla in pixel, spostamento longitudinale $\Delta x$, velocita' dell'attore in km/h) unitamente alle regole formali di classificazione cinematica, il comportamento del modello muta sensibilmente:
   - Su **21 clip su 50 (42.0%)**, la conoscenza di dominio convince il VLM ad abbandonare la predizione allucinata `Forward_Pass`, riconoscendo correttamente controlli orientati e conduzioni palla.
3. **Conclusioni Scientifiche per la Tesi (La Necessita' del Paradigma Neuro-Simbolico):**
   - L'arricchimento del prompt con regole di dominio (*Soft In-Context Prompting*) migliora sensibilmente la consapevolezza del modello neurale, rettificando il 42% delle allucinazioni.
   - Tuttavia, poiche' i modelli linguistici/multimodali sono predittori statistici di token e non motori aritmetici rigidi, essi falliscono nel verificare con costanza disuguaglianze numeriche complesse (es. $-506 < -100$).
   - Di conseguenza, l'integrazione di un **motore di regole neuro-simbolico deterministico ed esterno** (come quello proposto in questa tesi) si rivela **strettamente necessaria e insostituibile** per garantire un Event Log corretto al 100%, privo di rumore semantico e pienamente conforme agli standard del Process Mining.

---

## 6. Documenti e Tabelle Accademiche Generate

Tutti i risultati numerici e i modelli grafici sono disponibili nella cartella `results/` e pronti per l'inclusione nella stesura della tesi:
- `results/latex_table_50_clips_live_dual_qwen.tex`: Tabella formale in LaTeX con il confronto puntuale tra Zero-Shot e Con Knowledge Base per tutte le 50 clip.
- `results/latex_table_comparison.tex`: Tabella riassuntiva della tassonomia delle discrepanze sul dataset globale di 767 clip.
- `results/process_map_dfg.png`: Rappresentazione grafica del Directly-Follows Graph per il primo tempo di gioco.
- `results/pass_network_full_match.png`: Grafo orientato della Social Network Analysis dei passaggi.
- `results/activity_distribution_full_match.png`: Grafico della distribuzione delle macro-attivita' di gara.

---

## 7. Riferimenti Bibliografici

1. **van der Aalst, W. M. P.** (2016). *Process Mining: Data Science in Action*. Springer.
2. **Pettersen, S. A., Johansen, D., Johansen, H., Berg-Johansen, V., Gaddam, V. R., Mortensen, A., Langseth, R., Griwodz, C., Stensland, H. K., & Halvorsen, P.** (2014). *Alfheim: An open multi-sensor dataset from professional soccer matches*. Proceedings of the 5th ACM Multimedia Systems Conference (MMSys '14), pp. 200–205.
3. **Ferrara, A., et al.** (2024). *AA-STAL: Spatial-Temporal Action Localization with Visual Prompting and Set-of-Marks*. Computer Vision and Pattern Recognition.
4. **Qwen Team** (2025). *Qwen2.5-VL: Enhancing Vision-Language Capabilities for Dynamic and High-Resolution Multimodal Understanding*. arXiv preprint.
5. **Berti, A., et al.** (2019). *Process Mining for Python (PM4Py): An open-source library for process analysis*. CEUR Workshop Proceedings.
