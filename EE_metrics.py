from sacrebleu.metrics import BLEU
from bert_score import score
from nltk.translate.meteor_score import single_meteor_score
from sacrebleu.metrics import CHRF
import os
import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from Config import config
class EE_metric:
    def get_segment_scores(self,sys2H_result_dfs,dataset_name):
        raise NotImplementedError
    def get_corpus_score(self,hyps,refs):
        raise NotImplementedError
    def __init__(self): #backbone_type='seg' or 'corp'
        self.backbone_type=None
        self.cache_path=config['cache_path']
    def get_sys_scores(self,sys2H_result_dfs,dataset_name,threhold,diff_easy_coef,return_vallina_score=False):
        system_scores={}
        system_vallina_scores={}
        if self.backbone_type=='corp':
            for sys,df in sys2H_result_dfs.items():
                data_easy=df[df["H"].lt(threhold)]
                data_difficulty=df[df["H"].ge(threhold)]
                vallina_score=self.get_corpus_score(df["hyp"].to_list(),[df["ref"].to_list()])
                system_vallina_scores[sys]=vallina_score
                if not data_easy["hyp"].to_list():
                    easy_EE_score=0
                else:
                    easy_EE_score=self.get_corpus_score(data_easy["hyp"].to_list(),[data_easy["ref"].to_list()])
                if not data_difficulty["hyp"].to_list():
                    diff_EE_score=0
                else:
                    diff_EE_score=self.get_corpus_score(data_difficulty["hyp"].to_list(),[data_difficulty["ref"].to_list()])
                EE_score=diff_easy_coef*easy_EE_score+(1-diff_easy_coef)*diff_EE_score
                system_scores[sys]=EE_score
        if self.backbone_type=='seg':
            sys2segment_scores=self.get_segment_scores(sys2H_result_dfs,dataset_name)
            for sys,df in sys2H_result_dfs.items():
                system_vallina_scores[sys]=sys2segment_scores[sys]['scores'].mean()
                data_easy_index=df["H"].lt(threhold)
                data_difficulty_index=df["H"].ge(threhold)
                if not df[data_easy_index]["hyp"].to_list():
                    easy_EE_score=0
                else:
                    easy_EE_score=sys2segment_scores[sys]['score'].to_list()
                    easy_EE_score=sum(easy_EE_score)/len(easy_EE_score)
                if not df[data_difficulty_index]["hyp"].to_list():
                    diff_EE_score=0
                else:
                    diff_EE_score=sys2segment_scores[sys][data_difficulty_index]['scores'].to_list()
                    diff_EE_score=sum(diff_EE_score)/len(diff_EE_score)
                EE_score=diff_easy_coef*easy_EE_score+(1-diff_easy_coef)*diff_EE_score
                system_scores[sys]=EE_score
        assert len(system_scores)!=0
        if return_vallina_score:
            return system_scores,system_vallina_scores
        else:
            return system_scores

class EE_BLEU(EE_metric):
    def __init__(self):
        super().__init__()
        self.backbone_type='corp'
        self.bleu=BLEU()
    def get_corpus_score(self, hyps, refs):
        return self.bleu.corpus_score(hyps,refs).score

class EE_BertScore(EE_metric):
    def Bert_choice(self,dataset_name):
        if 'to_zh' in dataset_name:
            return config['chinese_bert_path'],config['chinese_bert_layer']
        if 'to_en' in dataset_name:
            return config['english_lan_bert_path'],config['english_bert_layer']
        return config['other_lan_bert_path'],config['other_lan_bert_path']
    def __init__(self):
        super().__init__()
        self.backbone_type='seg'
    def get_segment_scores(self, sys2H_result_dfs, dataset_name):
        #check cache
        cache_file_name=os.path.join(self.cache_path,"%s_%s.pkl"%(dataset_name,self.__class__.__name__))
        if os.path.exists(cache_file_name):
            with open(cache_file_name,'rb') as f:
                return pickle.load(f)
        else:
            sys2segment_scores={}
            for sys,df in tqdm(sys2H_result_dfs.items()):
                if 'zh_to_en' in dataset_name:
                    _,_,vallina_bertScore=score(df["hyp"].to_list(),df["ref"].to_list(),model_type="microsoft/deberta-xlarge-mnli",verbose=False,device='cuda:0',batch_size=32)
                else:
                    _,_,vallina_bertScore=score(df["hyp"].to_list(),df["ref"].to_list(),model_type=self.Bert_choice(dataset_name)[0],num_layers=self.Bert_choice(dataset_name)[1],verbose=False,device='cuda:0',batch_size=32)
                sys2segment_scores[sys]=pd.DataFrame(data=vallina_bertScore,columns=['scores'])
            # save cache
            with open(cache_file_name,'wb') as f:
                pickle.dump(sys2segment_scores,f)
            return sys2segment_scores

class EE_Meteor(EE_metric):
    def __init__(self):
        super().__init__()
        self.backbone_type='seg'
    def get_segment_scores(self, sys2H_result_dfs, dataset_name):
        #check cache
        cache_file_name=os.path.join(self.cache_path,"%s_%s.pkl"%(dataset_name,self.__class__.__name__))
        if os.path.exists(cache_file_name):
            with open(cache_file_name,'rb') as f:
                return pickle.load(f)
        else:
            sys2segment_scores={}
            for sys,df in tqdm(sys2H_result_dfs.items()):
                pre_tokenized_ref=[ele.split() for ele in df["ref"].to_list()]
                pre_tokenized_hyp=[ele.split() for ele in df["hyp"].to_list()]
                vallina_meteor_ls=[]
                for one_ref,one_hyp in zip(pre_tokenized_ref,pre_tokenized_hyp):
                    vallina_meteor_tmp=single_meteor_score(one_ref,one_hyp)
                    vallina_meteor_ls.append(vallina_meteor_tmp)
                sys2segment_scores[sys]=pd.DataFrame(data=vallina_meteor_ls,columns=['scores'])
            # save cache
            with open(cache_file_name,'wb') as f:
                pickle.dump(sys2segment_scores,f)
            return sys2segment_scores        


class EE_CHRF(EE_metric):
    def __init__(self):
        super().__init__()
        self.backbone_type='seg'
        self.chrf=CHRF()
    def get_segment_scores(self, sys2H_result_dfs, dataset_name):
        #check cache
        cache_file_name=os.path.join(self.cache_path,"%s_%s.pkl"%(dataset_name,self.__class__.__name__))
        if os.path.exists(cache_file_name):
            with open(cache_file_name,'rb') as f:
                return pickle.load(f)
        else:
            sys2segment_scores={}
            for sys,df in tqdm(sys2H_result_dfs.items()):
                chrf_ls=[]
                for hyp,ref in zip(df["hyp"].to_list(),df["ref"].to_list()):
                    chrf_ls.append(self.chrf.sentence_score(hyp,[ref]).score)
                sys2segment_scores[sys]=pd.DataFrame(data=chrf_ls,columns=['scores'])
            # save cache
            with open(cache_file_name,'wb') as f:
                pickle.dump(sys2segment_scores,f)
            return sys2segment_scores        