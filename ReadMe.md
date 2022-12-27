# EE Metric
EE Metric (Entropy Enhanced Metric) is built upon existing metrics, aiming to achieve a more balanced system-level rating by assigning weights to segment-level scores produced by backbone metrics. The weights are determined by the difficulty of a segment, which is related to the entropy of a hypothesis-reference pair.A translation hypothesis with a significantly high entropy value is considered difficult and receives a large weight in aggregation of EE Metrics’ system-level scores.
## Requirements
fast_align  
transformers
## Usage
(1) Fill in the paths in Config.py  
(2) Run example.py to get the correlations of WMT19
## Cite
Liu, Y., Tao, S., Su, C., Zhang, M., Zhao, Y., & Yang, H. (2022, November). Part represents whole: Improving the evaluation of machine translation system using entropy enhanced metrics. In Findings of the Association for Computational Linguistics: AACL-IJCNLP 2022 (pp. 296-307).  


