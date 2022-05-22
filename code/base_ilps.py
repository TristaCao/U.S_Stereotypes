from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import BertTokenizer, BertForMaskedLM

import torch
from torch.nn import functional as F

import numpy as np
import pandas as pd
import math
import statistics

def equal(l1,l2):
    assert len(l1) == len(l2)
    for x,y in zip(l1,l2):
        if x!=y: return False
    return True

def get_log_prob(mname, model, tokenizer, sentence, trait):
    trait_ids = tokenizer.encode(trait, return_tensors='pt').squeeze().tolist()
    trait_ids = trait_ids[1:len(trait_ids)-1]
    trait_ids = trait_ids[0:1]
    token_ids = tokenizer.encode(sentence, return_tensors='pt')
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position]
    with torch.no_grad(): output = model(token_ids)
    logits = output.logits.squeeze()
    masked_probs = []
    for idx, t_idx in zip(masked_pos, trait_ids):
        probs = torch.softmax(logits[idx],-1)
        prob = probs[t_idx].item()
        masked_probs.append(prob)
    log_prob = np.sum(np.log(np.array(masked_probs)))
    return log_prob

def bwoe(mname, model, tokenizer, groups, traits, prior_group, tmplt="The <group> is <mask>.", median=True):
    logging_file = 'results/{}_base_logging.txt'.format(mname)
    # tlogging_file = 'results/{}_base_logging_target.txt'.format(mname)
    # plogging_file = 'results/{}_base_logging_prior.txt'.format(mname)
    with open(logging_file,'a+') as lfile:
        lfile.write(f"{tmplt}===================================================")
        lfile.write("\n")
    # with open(tlogging_file,'a+') as lfile:
    #     lfile.write(f"{tmplt}===================================================")
    #     lfile.write("\n")
    # with open(plogging_file,'a+') as lfile:
    #     lfile.write(f"{tmplt}===================================================")
    #     lfile.write("\n")
    scores = {'traits': traits.keys()}
    for group in groups:
        group_scores = []
        dims = []
        for dim, ts in traits.items():
            dim_scores = []
            p_scores = []#
            t_scores = []#
            dims.append(dim)
            for trait in ts:
                if mname == 'roberta':
                    trait = ' '+trait
                trait_ids = tokenizer.encode(trait, return_tensors='pt').squeeze()
                trait_len = list(trait_ids.shape)[0]-2
                input_txt = tmplt.replace(' <mask>', (' '+tokenizer.mask_token)*trait_len).replace('<group>',prior_group)
                prior = get_log_prob(mname, model, tokenizer, input_txt, trait)
                input_txt = tmplt.replace(' <mask>', (' '+tokenizer.mask_token)*trait_len).replace('<group>',group)
                target = get_log_prob(mname, model, tokenizer, input_txt, trait)
                lps_score = target-prior
                p_scores.append(prior)#
                t_scores.append(target)#
                dim_scores.append(lps_score)
            with open(logging_file,'a+') as lfile:
                ss = [str(x) for x in dim_scores]
                score_string = '\t'.join(ss)
                lfile.write(f"{group}\t{dim}\t{score_string}")
                lfile.write("\n")
            # with open(plogging_file,'a+') as lfile:
            #     p_scores = [str(x) for x in p_scores]
            #     score_string = '\t'.join(p_scores)
            #     lfile.write(f"{group}\t{dim}\t{score_string}")
            #     lfile.write("\n")
            # with open(tlogging_file,'a+') as lfile:
            #     t_scores = [str(x) for x in t_scores]
            #     score_string = '\t'.join(t_scores)
            #     lfile.write(f"{group}\t{dim}\t{score_string}")
            #     lfile.write("\n")

            if median: group_scores.append(np.median(np.array(dim_scores)))
            if not median: group_scores.append(np.mean(np.array(dim_scores)))   
            
        scores[group] = group_scores
        
        assert equal(traits.keys(), dims)
        df = pd.DataFrame(data=scores)
    return df



def main():
    mname = 'roberta'
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-large',cache_dir="../../hf_cache_dir/")
    # model = RobertaForMaskedLM.from_pretrained('roberta-large',cache_dir="../../hf_cache_dir/")
    # # mname = 'bert'
    # # tokenizer = BertTokenizer.from_pretrained('bert-large-cased',cache_dir="../hf_cache_dir/")
    # # model = BertForMaskedLM.from_pretrained('bert-large-cased',cache_dir="../hf_cache_dir/")
    # model.eval()
    # traits = {'powerless':['powerless','weak','helpless','incapable'],
    #     'powerful':['powerful','forceful','formidable','capable'],
    #     'low-status':['low-status','unskilled','lowly','inferior'],
    #     'high-status':['high-status','privileged','elite','advantaged'],
    #     'dominated':['dominated','subservient','submissive','deferential'],
    #     'dominant':['dominant','commanding','authoritative'],
    #     'poor':['poor','impoverished','destitute','needy'],
    #     'wealthy':['wealthy','affluent','rich','prosperous'],
    #     'unconfident':['unconfident','bashful','meek','timid'],
    #     'confident':['confident','self-assured','assured','self-possessed'],
    #     'unassertive':['unassertive','submissive','diffident','passive'],
    #     'competitive':['competitive','ambitious','driven','zealous'],
    #     'traditional':['traditional','old-fashioned'],      
    #     'modern':['modern','radical','forward-looking'],    
    #     'religious':['religious','devout','pious','reverent'],
    #     'science-oriented':['science-oriented','analytical','logical','athiestic'],
    #     'conventional':['conventional', 'mainstream'],      
    #     'alternative':['alternative','unorthodox','avante-garde','eccentric'],
    #     'conservative':['conservative','right-wing','Republican'],  
    #     'liberal':['liberal','left-wing','Democrat','progressive'],
    #     'untrustworthy':['untrustworthy','unreliable','undependable'],  
    #     'trustworthy':['trustworthy','reliable','dependable','truthful'],
    #     'dishonest':['dishonest','insincere','deceitful'],  
    #     'sincere':['sincere','genuine','forthright','honest'],
    #     'cold':['cold','unfriendly','unkind','aloof'],
    #     'warm':['warm','friendly','kind','loving'],
    #     'threatening':['threatening','intimidating','menacing','frightening'],
    #     'benevolent':['benevolent','considerate','generous'],   
    #     'repellent':['repellent','vile','loathsome','nasty'],
    #     'likable':['likable','pleasant','amiable','lovable'],
    #     'egotistic':['egotistic','selfish','self-centered','insensitive'],
    #     'altruistic':['altruistic','helpful','charitable','selfless']
    # }
    # groups = ["Hispanics","Asians","Immigrants","Gays","Blacks","African Americans"]
    # template = "<group> are more <mask> than others."
    # prior_group = 'People'
    # result=woe(mname, model, tokenizer, groups, traits, prior_group, template)
    # # result.to_csv(mname+'/baseline_pilot_median.csv')


if __name__ == "__main__":
	main()