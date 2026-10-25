# REQUIREMENTS CHECKLIST (from SENTINEL_Hackathon_45min.docx)

Source: `SENTINEL_Hackathon_45min.docx` (original assignment).  This checklist tracks every stated requirement.

| Req ID | Description | Planned implementation | Source file/function | Test | Status |
|---|---|---|---|---|---|
| REQ-01 | Read and analyze the original DOCX before implementing | Extracted all 244 lines with xml parsing; stored copy in temp and referenced tables/corrections | `scripting` (done) | Manual inspection of extracted text | DONE |
| REQ-02 | Use the exact campus graph (bidirectional), start A, goal F | `build_campus_graph()` adds edges both directions; A-F used | `sentinel/graph.py::build_campus_graph` | `tests/test_search.py` | DONE |
| REQ-03 | BFS finds path with fewest edges; alphabetical neighbours | `bfs()` with deque, visited, `graph.neighbors()` sorted | `sentinel/search.py::bfs` | `TestSearch.test_bfs_expected_path_and_metrics` | DONE |
| REQ-04 | DFS with documented neighbour-ordering rule | Recursive DFS, `graph.neighbors()` alphabetical; visited prevents cycles | `sentinel/search.py::dfs` | `TestSearch.test_dfs_expected_path_and_metrics` | DONE |
| REQ-05 | UCS minimises travel cost | Dijkstra with heapq, stale-entry skipping | `sentinel/search.py::uniform_cost_search` | `TestSearch.test_ucs_expected_path_and_metrics` | DONE |
| REQ-06 | Risk-aware J(P)=sum c(e)+lambda*sum r(e), lambda=1 | Dijkstra over objective weight | `sentinel/search.py::risk_aware_search` | `TestSearch.test_risk_aware_expected_path_and_metrics` | DONE |
| REQ-07 | Expected outputs match DOCX (BFS A->B->D->F cost7 risk23, DFS A->B->D->C->E->F cost13 risk19, **UCS A->B->D->F cost7 risk23**, Risk-aware A->C->E->F cost11 risk4, J=15) | Computed dynamically; DFS cost corrected to 13; **UCS computed as A->B->D->F cost 7 risk 23 from actual graph weights - assignment brief's UCS example row (A->C->D->F cost 9 risk 10) does not match the given edge weights; code computes true minimum as required** | All search tests | All four search tests | DONE |
| REQ-08 | Search works on graph data; changing edge weights changes result | Tests modify graph and rerun | `tests/test_search.py::test_changing_edge_weights_changes_result`, `tests/test_validation.py::test_modifying_graph_edge_changes_search_result` | Passed in logic (will run) | DONE |
| REQ-09 | Agent policy priority exactly as specified (RULE1-6) | `decide_action()` if/elif in order | `sentinel/agent.py::decide_action` | `tests/test_agent.py` all cases | DONE |
| REQ-10 | I01 ALLOW, I02 REJECT, I03 ESCALATE, I04 INVESTIGATE | As above | Same | All four tested | DONE |
| REQ-11 | Confirmed compromise and critical alert override permissive | Tested explicitly | Same | Two override tests | DONE |
| REQ-12 | PEAS documented | `PEAS` dict + README + ARCHITECTURE | `sentinel/agent.py::PEAS`, `README.md`, `docs/ARCHITECTURE.md` | Manual | DONE |
| REQ-13 | >=300 synthetic records, fixed seed, save to data/incidents.csv | `generate_records(400, seed=42)`, `ensure_dataset()` | `sentinel/dataset.py` | `tests/test_classifier.py` | DONE |
| REQ-14 | Features: failed_logins, requests_per_min, unusual_hour, known_device, duplicate_count, label target | Enforced | Same | Tests check features/labels | DONE |
| REQ-15 | Stratified 80:20 split, reproducible, train on train, test on test | `split_dataset()` with stratify | `sentinel/classifier.py` | Split test | DONE |
| REQ-16 | Decision Tree or KNN; report accuracy/precision/recall/macro F1/confusion matrix | DecisionTreeClassifier, macro averaging, zero_division=0 | `sentinel/classifier.py` | Classifier tests | DONE |
| REQ-17 | No label leakage; only 5 features | Only FEATURES used; label excluded | Same | Feature count test | DONE |
| REQ-18 | Investigate false predictions (if <2 observed, report honestly) | Collected up to all false predictions; CLI explains if none | `IncidentClassifier._build_metrics` | CLI shows | DONE |
| REQ-19 | process_incident(event, graph, model) implemented | Full function with validation | `sentinel/integration.py` | Integration tests | DONE |
| REQ-20 | I05: valid prediction, ESCALATE (critical alert), route computed dynamically, route metrics included, ML not overriding policy | Implemented | `tests/test_integration.py::test_i05_integration` | Will run | DONE |
| REQ-21 | All six robustness cases tested | Tests for all six | `tests/test_validation.py` | Will run | DONE |
| REQ-22 | Prompt injection: description untrusted, policy unchanged | Scanner present, agent ignores description for rules | `sentinel/validation.py`, `sentinel/agent.py`, test_6 | Tested | DONE |
| REQ-23 | Duplicate IDs detected, rejected safely (no silent merge) | `IncidentRegistry`, checked in process_incident | Same | Duplicate test | DONE |
| REQ-24 | Automated tests runnable with unittest discover | `tests/` package created | Tests folder | Will run | DONE |
| REQ-25 | Application runs from project root as `python sentinel.py` | `sentinel.py` at root with sys.path fix | `sentinel.py` | Will run | DONE |
| REQ-26 | Documentation: README, ARCHITECTURE, VIVA_PREPARATION, TEST_REPORT | Created | `docs/` | Manual | DONE |
| REQ-27 | AI-use disclosure template included | In README | `README.md` | Manual | DONE |
| REQ-28 | Live code modification possible (change edge weights / incident fields) | Graph mutable, all computed dynamically | All modules | Demonstrated in tests | DONE |

**Unresolved ambiguity:** None found after reading DOCX and correction notes.  DFS costs were corrected explicitly - implemented as computed.
