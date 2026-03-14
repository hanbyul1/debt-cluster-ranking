# Debt Cluster Ranking

This repository contains the experimental artifacts used in the study:

**Topology-Aware Technical Debt Identification and Ranking in LLM-Assisted Software Development**

The repository provides system snapshots and evaluation scripts used to analyze and compare topology-aware technical debt clustering with a baseline metric-based prioritization approach.

---

## Repository Structure

```
experiments/
│
├── case systems/
│   ├── MenuCal/
│   │   └── snapshot trees/
│   │        ├── MenuCal_D1_swift.txt
│   │        ├── ...
│   │        └── MenuCal_D12_swift.txt
│   │
│   └── NutriCompass/
│        └── snapshot trees/
│             ├── NutriCompass_D1_swift.txt
│             ├── ...
│             └── NutriCompass_D12_swift.txt
│
└── evaluation/
     ├── baseline_debt_analysis.py
     ├── topology_debt_analysis.py
     ├── comparison_pipeline.py
     │
     ├── baseline type analysis/
     ├── proposed cluster analysis/
     └── comparison results/
```

---

## Case Systems

Two AI-assisted software systems are analyzed in this study:

- **MenuCal** – an on-device AI calorie estimation application.
- **NutriCompass** – an evidence-indexed supplement analysis system.

Each system contains **12 development snapshots (D1–D12)** capturing the evolution of the codebase during LLM-assisted development.

---

## Snapshot Trees

Snapshot files represent extracted Swift source trees for each development stage.

Example:

```
MenuCal_D1_swift.txt
MenuCal_D2_swift.txt
...
MenuCal_D12_swift.txt
```

These snapshots are used to construct:

- dependency graphs  
- call graphs  
- structural maintainability metrics  
- technical debt clusters  

The snapshots enable longitudinal analysis of structural evolution across development iterations.

---

## Evaluation Scripts

The evaluation scripts are located under:

```
experiments/evaluation/
```

Key scripts include:

- **baseline_debt_analysis.py**  
  Implements the baseline prioritization approach using maintainability metrics  
  (CC, WMT, LCOT, CBT, RFT).

- **topology_debt_analysis.py**  
  Implements the proposed topology-aware technical debt clustering and ranking approach.

- **comparison_pipeline.py**  
  Generates comparative results between the baseline and topology-aware approaches across system snapshots.

The scripts operate on the snapshot trees located under:

```
experiments/case systems/
```

---

## Development Chat Logs

Development chat sessions with LLMs are provided through the **maintainability repository**, which is linked as a Git submodule:

```
experiments/case systems/maintainability
```

This repository contains development chat session logs for both case systems.

The submodule is pinned to a specific commit to ensure artifact reproducibility.

---

## Experimental Goal

The experiments evaluate a **topology-aware technical debt identification and ranking framework** that:

1. Detects debt-bearing types using maintainability metrics  
2. Constructs dependency graphs between system types  
3. Identifies structurally cohesive debt clusters  
4. Ranks clusters to support technical debt prioritization  

The approach is evaluated across **longitudinal development snapshots** of both case systems.

---

## Reproducibility

All experiments are conducted using the snapshot trees provided in this repository.

The maintainability repository provides development context through LLM interaction logs.

Together, these artifacts support reproducibility of the structural analysis and prioritization results reported in the study.

---

## License

This repository is provided for research and artifact evaluation purposes.
