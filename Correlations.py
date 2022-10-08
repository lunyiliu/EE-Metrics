import itertools
from scipy.stats import pearsonr,spearmanr
from scipy.stats import kendalltau
from random import sample
import itertools
def compute_correlations(system_scores,sys2da):
    pred_scores=[system_scores[sys] for sys in sys2da]
    GT_scores=[sys2da[sys] for sys in sys2da]
    kendall=kendalltau(GT_scores,pred_scores)[0]
    pearson=pearsonr(GT_scores,pred_scores)[0]
    spearman=spearmanr(GT_scores,pred_scores)[0]
    return pearson,spearman,kendall
