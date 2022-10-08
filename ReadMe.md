# EE Metric
EE Metric (Entropy Enhanced Metric) is built upon existing metrics, aiming to achieve a more balanced system-level rating by assigning weights to segment-level scores produced by backbone metrics. The weights are determined by the difficulty of a segment, which is related to the entropy of a hypothesis-reference pair.A translation hypothesis with a significantly high entropy value is considered difficult and receives a large weight in aggregation of EE Metrics’ system-level scores.
## Requirements
fast_align  
transformers
## Usage
(1) Fill in the paths in Config.py  
(2) Run example.py to get the correlations of WMT19
## Cite
TBA