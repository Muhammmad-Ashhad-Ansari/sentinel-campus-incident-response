# ARCHITECTURE

## High-level data flow

```
INPUT: events (I01-I05 + custom)  |  campus graph (A-F, bidirectional)
                |                                |
                v                                v
            validation.py                    graph + search.py
                |                                (BFS/DFS/UCS/Risk)
                v
            process_incident()  --predicts-->  classifier.py (3-class)
                |                              (5 features, no leakage)
                v
            decide_action()  (agent.py)  --policy rules RULE_1-6--
                |                              (policy authoritative)
                v
            JSON result: predicted_class, action, triggered_rule,
                          explanation, evidence, route (if requested)
```

## Module responsibilities

| Module | Purpose |
|---|---|
| sentinel/graph.py | Validated undirected graph; cost and risk per edge; rejects negative cost and risk outside 0-10; alphabetically-ordered neighbours |
| sentinel/search.py | BFS (fewest edges), DFS (alphabetical, cycle-safe), UCS/Dijkstra (min cost), Risk-aware Dijkstra (min J = sum cost + lambda*sum risk). Metrics computed from path+graph |
| sentinel/agent.py | Rule-based agent with mandatory priority order RULE_1-6. PEAS model. Returns structured decision with rule and explanation |
| sentinel/validation.py | Field/range validation, contradiction detection, duplicate-ID tracking, prompt-injection scanning (evidence only) |
| sentinel/dataset.py | Deterministic generator (seed 42), writes data/incidents.csv, includes 12 starter rows + generated rows to reach >=300 |
| sentinel/classifier.py | DecisionTreeClassifier, stratified 80:20 split, macro precision/recall/F1, confusion matrix, false-prediction analysis, no label leakage |
| sentinel/integration.py | process_incident(event, graph, model) - validates, predicts, applies agent policy (authoritative), computes route if requested, returns JSON-compatible dict |
| sentinel/reporting.py | Console formatting helpers |
| sentinel.py | CLI demo orchestrating all components |

## Design principles

- Graph-first: search never hardcodes answers; everything computed from CampusGraph.
- Policy-first: the agent's priority rules are authoritative. The ML classifier never overrides them.
- Separation of concerns: prediction (ML) and action selection (agent) are different responsibilities.
- Validation over trust: all inputs validated; invalid/malicious inputs fail safely.
- Reproducibility: fixed seeds, deterministic generation, stratified split.
- Testability: pure functions, small modules, comprehensive unittest suite.
- Explainability: every decision includes triggered rule, explanation, and evidence.
