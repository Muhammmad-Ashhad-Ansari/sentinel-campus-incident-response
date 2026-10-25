"""SENTINEL - Intelligent Campus Incident Response Agent.

This package contains small, separate modules for the four assignment tasks:

* ``graph``       - the campus graph and its validation rules
* ``search``      - BFS, DFS, UCS and risk-aware search
* ``agent``       - the six-rule decision agent and the PEAS model
* ``dataset``     - deterministic synthetic incident-data generation
* ``classifier``  - the scikit-learn decision-tree classifier
* ``validation``  - input, contradiction, duplicate-ID and injection checks
* ``integration`` - process_incident(event, graph, model)
* ``reporting``   - small console/table helpers
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
