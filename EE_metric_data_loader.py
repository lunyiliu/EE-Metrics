import os
import math
import re
from typing import list
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
import pickle
from scipy.stats import norm
from Config import config
class EE_metric_data_loader:
    def __init__(self,systems_output_path,ref_file,dataset_name,excluded_sys_names=[],da_path=None,is_chinese=False,da_type='xlsx',sys_name_type=19,da_delimiter='\t'):
        sys2H_result_dfs_cache_file="cache/sys2H_result_dfs_%s.pkl"%dataset_name
        if os.path.exists(sys2H_result_dfs_cache_file):
            with open(sys2H_result_dfs_cache_file,'rb') as f:
                self.sys2H_result_dfs=pickle.load(f)
        else:
            list_systems_output_path=[ele for ele in os.listdir(systems_output_path) if '.ipynb_checkpoints' not in ele and '_tokenized' not in ele]
            if sys_name_type==19:
                system_names=['.'.join(path.split('.')[1:-1]) for path in list_systems_output_path]
            else:
                system_names=['.'.join(path.split('.')[:-1]) for path in list_systems_output_path]
            system_outputs=[]
            for i in range(0,len(list_systems_output_path)):
                system_outputs.append(os.path.join(systems_output_path,list_systems_output_path[i]))
            self.is_chinese=is_chinese
            if self.is_chinese:
                self.tokenize_chinese(ref_file)
                for system_output,name in zip(system_outputs,system_names):
                    self.tokenize_chinese(system_output)
            self.sys2H_result_dfs={}
            for system_output,name in tqdm(zip(system_outputs,system_names)):
                self.sys2H_result_dfs[name]=self.get_H(system_output,ref_file,output_log=False)
            with open(sys2H_result_dfs_cache_file,'wb') as f:
                pickle.dump(self.sys2H_result_dfs,f)
        if da_path:
            if da_type=='xlsx':
                da=pd.read_excel(da_path,engine='openpyxl')
                self.sys2da={}
                for i in range(len(da)):
                    text=da.loc[i,'text'].split(' ')
                    self.sys2da[text[-1]]=float(text[1])
            else:
                self.sys2da={}
                with open(da_path) as f:
                    for line in f.readlines():
                        system,score=line.split(da_delimiter)
                        self.sys2da[system]=float(score)
        self.effective_sys_names=[sys for sys in self.sys2H_result_dfs if sys in self.sys2da and sys not in excluded_sys_names]
        self.avg_H_ls=[]
        for sys in self.effective_sys_names:
            df=self.sys2H_result_dfs[sys]
            if self.avg_H_ls==[]:
                self.avg_H_ls=np.asarray(df["H"].to_list())
            else:
                self.avg_H_ls+=np.asarray(df["H"].to_list())
        self.avg_H_ls/=len(self.sys2H_result_dfs)
        self.avg_H_ls=[ele for ele in self.avg_H_ls if ele!=0]
    def tokenize_chinese(self,file):
        data_ls=[]
        with open(file,'r') as f:
            f.write(''.join(data_ls))
    def get_ent(self,hypotheses,references,aligns):
        assert len(hypotheses) == len(references) == len(aligns)
        Hs=[]
        for index,i in enumerate(aligns):
            if self.is_chinese:
                hyp=' '.join(list(hypotheses[index].strip()))
                ref=' '.join(list(references[index].strip()))
            else:
                hyp=re.sub(" +"," ",hypotheses[index].strip())
                ref=re.sub(" +"," ",references[index].strip())
            align=i.strip().replace("-"," ").split(" ")
            align=list(set([int(x) for x in align[1::2]]))
            if not align:
                l_h=len(hyp.split(" "))
            else:
                l_h=max(max(align)+1,len(hyp.split(" ")))
            l_r=len(ref.split(" "))
            plate=np.array(["0"]*l_h)
            plate[align]="1"
            data=[len(x) for x in "".join(plate.tolist()).split("0") if len(x)>0]
            l_i=np.array(data)
            l=l_i/l_i.sum()
            H=-((l*np.log10(l)).sum())
            Hs.append(H)
        return Hs
    def get_align(self,CORPUS_A,CORPUS_B):
        if self.is_chinese:
            CORPUS_A+='_tokenized'
            CORPUS_B+='_tokenized'
        align_file_name="cache/align_%s_to_%s"%(CORPUS_B.split(r"/")[-1],CORPUS_A.split(r"/")[-1])
        fast_align=config['fast_align_path']
        os.system("paste -d '`' %s %s > tmp_b"%(CORPUS_B,CORPUS_A))
        os.system("sed -i 's/`/ ||| /g' tmp_b")
        os.system("%s -i tmp_b -d -o -v > %s"%(fast_align,align_file_name))
        os.system("rm tmp_b")
        return align_file_name
    def get_H(self,CORPUS_A,CORPUS_B,output_log=False):
        align_file_name=self.get_align(CORPUS_A,CORPUS_B)
        output_file_name="cache/tgt-%s_to_ref-%s.csv"%(CORPUS_A.split(r"/")[-1],CORPUS_B.split(r"/")[-1])
        with open(CORPUS_A,'r') as f: # hypo file A
            hypotheses=f.readlines()
            hypotheses=[re.sub('[`“”;,.‘"]','',i).strip() for i in hypotheses]
        with open(CORPUS_B,'r') as f: # ref file B
            references=f.readlines()
            references=[re.sub('[`“”;,.‘"]','',i).strip() for i in references]
        with open (align_file_name,'r') as f:
            aligns=f.readlines()
            aligns=[i.strip() for i in aligns]
        Hs=self.get_ent(hypotheses,references,aligns)
        result=pd.DataFrame({'hyp':hypotheses,'ref':references,'H':Hs,'align':aligns})
        if output_log:
            result.to_csv(output_file_name,index=False,encoding='utf-8')
        os.system("rm %s"%align_file_name)
        return result
    def get_threshold(self):
        mu,std=norm.fit(self.avg_H_ls)
        return mu+2*std
    def get_w(self,threshold):
        easy_num=sum([1 for ele in self.avg_H_ls if ele<=threshold])
        diff_num=sum([1 for ele in self.avg_H_ls if ele>threshold])
        easy_sum=sum([ele for ele in self.avg_H_ls if ele<=threshold])
        diff_sum=sum([ele for ele in self.avg_H_ls if ele>threshold])
        R_H=easy_sum/diff_sum
        R_N=easy_num/diff_num
        w=R_N/(9.62*R_H+R_N-22.23)
        return w