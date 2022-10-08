import os
from EE_metric_data_loader import EE_metric_data_loader
from EE_metrics import EE_BLEU,EE_CHRF,EE_Meteor,EE_BertScore
from tqdm import tqdm
from Correlations import compute_correlations 
from Config import config
da_delimiter=' '
da_type='xlsx'
sys_name_type=19
ref_number='A'
ROOT_path=config['data_root_path']
#WMT19
lan_pairs=['en-de','de-en','en-zh','zh-en','en-gu']
excluded_sys_names=[]
dataset_name2lan_pair=[]
dataset_name2paths={'newstest2019_%s_to_%s'%(lan.split('-')[0],lan.split('-')[1]):
[os.path.join(ROOT_path,'%s-%s-sys'%(lan.split('-')[0],lan.split('-')[1])),
os.path.join(ROOT_path,'newstest2019-%s%s-ref.%s'%(lan.split('-')[0],lan.split('-')[1],lan.split('-')[1])),
os.path.join(ROOT_path,'DA-%s%s.xlsx'%(lan.split('-')[0],lan.split('-')[1]))] for lan in lan_pairs}
ee_metrics={'EE_BertScore':EE_BertScore(),'EE_BLEU':EE_BLEU(),'EE_CHRF':EE_CHRF(),'EE_Meteor':EE_Meteor()}
def get_correlations(ee_metrics,excluded_sys,diff_easy_coef=None,random_K=None,pair_wise=False):
    results={}
    correlations={}
    correlations_vallina={}
    correlations_delta={}
    threholds={'newstest2019_en_to_de':0.53,'newstest2019_de_to_en':0.52,'newstest2019_en_to_zh':0.84,'newstest2019_zh_to_en':0.76,'newstest2019_en_to_gu':0.72}
    for dataset_name,paths in dataset_name2paths.items():
        if 'to_zh' in dataset_name:
            is_chinese=True
        else:
            is_chinese=False
        data_loader=EE_metric_data_loader(paths[0],paths[1],dataset_name,da_path=paths[2],excluded_sys_names=excluded_sys_names,is_chinese=is_chinese,da_type=da_type,sys_name_type=sys_name_type,da_delimiter=da_delimiter)
        result={}
        correlation={}
        correlation_vallina={}
        #correlation_delta={}
        #threholds[dataset_name]=data_loader.get_threshold()
        if not diff_easy_coef:
            w=data_loader.get_w(threholds[dataset_name])
        else:
            w=diff_easy_coef
        for metric_name,ee_metric in ee_metrics.items():
            system_scores,system_vallina_scores=ee_metric.get_sys_scores(data_loader.sys2H_result_dfs,dataset_name,threhold=threholds[dataset_name],diff_easy_coef=w,return_vallina_score=True)
            result[metric_name]=system_scores
            correlation[metric_name]=compute_correlations(result[metric_name],{sys:human_score for sys,human_score in data_loader.sys2da.items() if sys not in excluded_sys_names})
            correlation_vallina[metric_name]=compute_correlations(system_vallina_scores,{sys:human_score for sys,human_score in data_loader.sys2da.items() if sys not in excluded_sys_names})
            #correlation_delta[metric_name]=[score-score_vallina for score,score_vallina in zip(correlation[metric_name],correlation_vallina[metric_name])]
        results[dataset_name]=result
        correlations[dataset_name]=correlation
        correlation_vallina[dataset_name]=correlation_vallina
        #correlations_delta[dataset_name]=correlation_delta
    return correlations,correlation_vallina

correlations,correlation_vallina=get_correlations(ee_metrics,excluded_sys_names,diff_easy_coef=0.3)