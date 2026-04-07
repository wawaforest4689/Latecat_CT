import torch
from CodonTransformer.CodonPrediction import predict_dna_sequence
from CodonTransformer.CodonUtils import IterableJSONData,MAX_LEN,NUM_ORGANISMS,STOP_SYMBOL
from CodonTransformer.CodonData import prepare_data_from_fasta_for_infer,ORGANISM2ID2,ID2ORGANISM2
from CodonTransformer.CodonJupyter import format_model_output
from BBMLMpretrain_latefusion import plTrainHarness,BertTokenizer,MyBigBirdModel,BigBirdConfig,MAX_LR,WARM_UP,\
    CustomBigBirdEmbeddings,MyCodonBBMLMHead,MyCodonDecoderHead,LateFusionHead

import CAI
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evaluation import generate_privileged_CFT,dtw_distance,Jaccard_Coeff,codon_bp_similarity
import RNA

from datetime import datetime
import time


CODON_TABLE_DS=IterableJSONData('dataset/training_data.jsonl')


def inference_from_str(plmodel_path,ref_aa,att_type='block_sparse',deterministic=True,temperature=1,
              top_p=0.95,num_sequences=1):
    if not isinstance(ref_aa,list) or not isinstance(ref_aa[0],tuple) or not isinstance(ref_aa[0][0],str)\
            or not isinstance(ref_aa[0][1],str):
        raise ValueError("ref_aa should be of type list[tuple(str,str)].")

    valid_att_types = ['original_full', 'block_sparse']
    if att_type not in valid_att_types:
        raise ValueError(f"attention type should be within {valid_att_types}.")

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('tokenizing')
    config = BigBirdConfig(
        vocab_size=len(tokenizer),
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=MAX_LEN,
        type_vocab_size=NUM_ORGANISMS,
        sep_token_id=2,
        block_size=32,
    )
    model = MyBigBirdModel(config=config, kp=1, scale=1, asymptotic=False, step=1, type_lf=0.01)
    pl_model = plTrainHarness.load_from_checkpoint(checkpoint_path=plmodel_path,model=model,tokenizer=tokenizer,learning_rate=MAX_LR,
                                                   warmup_fraction=WARM_UP,config=config,valid_interval=1000,accumulation_steps=8)

    model=pl_model.model
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    predicts = []
    pred_dnas=[]
    organisms=[]
    for item in ref_aa:
        organism_name = item[0]
        protein = item[1]
        pred = predict_dna_sequence(protein=protein, organism=organism_name, device=device, tokenizer=tokenizer,
                                    model=model, attention_type=att_type, deterministic=deterministic,
                                    temperature=temperature, top_p=top_p, num_sequences=num_sequences,
                                    match_protein=True)

        if isinstance(pred, list):
            for p in pred:
                predicts.append(format_model_output(p))
                pred_dnas.append(p.predicted_dna)
        else:
            predicts.append(format_model_output(pred))
            pred_dnas.append(pred.predicted_dna)

        for i in range(num_sequences):
            organisms.append(organism_name)
    cais=cal_cai(CODON_TABLE_DS,pred_dnas,organisms,11)
    assert(len(cais)==num_sequences*len(ref_aa))
    for i in range(len(predicts)):
        predicts[i]+=f'CAI:{cais[i]}\n\n\n'

    print('predictions:',''.join(predicts))
    return predicts



def inference(plmodel_path,dataset,size=-1,att_type='block_sparse',deterministic=True,temperature=0.5,
              top_p=0.9,num_sequences=1,privil=True):
    if not isinstance(dataset,IterableJSONData):
        raise ValueError("dataset should be of type IterableJSONData.")
    valid_att_types=['original_full','block_sparse']
    if att_type not in valid_att_types:
        raise ValueError(f"attention type should be within {valid_att_types}.")
    if deterministic:
        num_sequences=1

    list_ds=list(dataset)
    print(f"length of testing dataset:{len(list_ds)}")
    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('tokenizing')
    config = BigBirdConfig(
        vocab_size=len(tokenizer),
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=MAX_LEN,
        type_vocab_size=NUM_ORGANISMS,
        sep_token_id=2,
        block_size=32,
    )
    model = MyBigBirdModel(config=config,organism_embed_dim=32,kp=1, scale=1, asymptotic=False, step=1, type_lf=0)
    pl_model = plTrainHarness(model=model,tokenizer=tokenizer,learning_rate=MAX_LR,
                                warmup_fraction=WARM_UP,config=config,valid_interval=2000,accumulation_steps=8)

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    check_point=torch.load(plmodel_path,map_location='cuda:0' if torch.cuda.is_available() else 'cpu',weights_only=False)
    pl_model.load_state_dict(check_point['state_dict'])

    model=pl_model.model
    model.eval()

    df={'id':[],'RefSeq_aa':[]}
    t_start=time.time()
    for item in list_ds:
        organism_name=item["organism"]
        protein=item.get("protein")
        if not item.get("protein"):
            organism_name=ID2ORGANISM2.get(organism_name)
            if len(df.get(organism_name+'_REF',[]))>=size:
                continue
            codons=item.get("codons").split(" ")
            protein="".join([c.split(STOP_SYMBOL)[0] for c in codons]).upper()
            dna = "".join([c.split(STOP_SYMBOL)[-1] for c in codons])
            df.setdefault(organism_name + '_REF', []).append(dna)
        else:
            if len(df.get(organism_name,[]))>=size*num_sequences:
                continue
        pred=predict_dna_sequence(protein=protein,organism=organism_name,device=device,tokenizer=tokenizer,
                             model=model,attention_type=att_type,deterministic=deterministic,
                             temperature=temperature,top_p=top_p,num_sequences=num_sequences,match_protein=True)

        if isinstance(pred, list):
            pred = [p.predicted_dna for p in pred]
        else:
            pred=pred.predicted_dna

        if protein not in df['RefSeq_aa']:
            print(item["idx"])
            df['id'].append(item["idx"])
            df['RefSeq_aa'].append(protein)
        df.setdefault(organism_name,[]).append(pred)

    duration=time.time()-t_start
    for k,v in df.items():
        if len(v)<len(df['RefSeq_aa']):
            df[k]=v+["" for i in range(len(df["RefSeq_aa"])-len(v))]
        print('Before length padding:','{: <40}{: >5}'.format(k,len(v)))
        print('After length padding: ','{: <40}{: >5}'.format(k,len(df[k])))

    df=pd.DataFrame(df)
    df.to_excel('Tests.xlsx',index=False, header=True)
    cais,gcs=[],[]
    ref_cais,ref_gcs=[],[]
    dtw_ds=[]
    mfe_diffs=[]
    jacs=[]
    cs,bps=[],[]

    w_b_o=None
    if privil:
        w_b_o=generate_privileged_CFT('dataset')

    for k in ORGANISM2ID2.keys():
        if num_sequences>1:
        # if isinstance(df[k].tolist()[0], list):
            # 所有推理序列都进行计算
            test_seqs=[]
            for seqs in df[k].tolist():
                if seqs=="":
                    break
                # 只计算第一条序列
                # test_seqs.append(seqs[0])
                for s in seqs:
                    test_seqs.append(s)
            cai,w_b_o=cal_cai(CODON_TABLE_DS, test_seqs, [k for i in range(len(test_seqs))], 11,weights_by_org=w_b_o)
            cais.append(cai)
            gcs.append(cal_gc(test_seqs,[k for i in range(len(test_seqs))]))
        else:
            test_seqs=df[k].tolist()
            try:
                empty_fp=test_seqs.index('')
                test_seqs=test_seqs[:empty_fp]
            except:
                pass
            cai,w_b_o=cal_cai(CODON_TABLE_DS, test_seqs, [k for i in range(len(test_seqs))], 11,weights_by_org=w_b_o)
            cais.append(cai)
            gcs.append(cal_gc(test_seqs,[k for i in range(len(test_seqs))]))

        if k+"_REF" in list(df.columns):
            ref_seqs=df[k+"_REF"].tolist()
            # 指定size里面只含有单一物种的情形
            try:
                empty_fp=ref_seqs.index('')
                ref_seqs=ref_seqs[:empty_fp]
            except:
                pass
            ref_cai,_=cal_cai(CODON_TABLE_DS, ref_seqs, [k for i in range(len(ref_seqs))], 11,w_b_o)
            ref_cais.append(ref_cai)
            ref_gcs.append(cal_gc(ref_seqs,[k for i in range(len(ref_seqs))]))

            # dtw/mfe/Jaccard/相似度计算
            assert(len(test_seqs)==num_sequences*len(ref_seqs))
            org_dtw_ds=[]
            org_mfe_diffs=[]
            org_jacs=[]
            org_cs=[]
            org_bps=[]
            for i in range(len(ref_seqs)):
                if 'UNK' in ref_seqs[i]:
                    continue
                for j in range(num_sequences):
                    dtw_d=dtw_distance(w_b_o,k,ref_seqs[i],test_seqs[num_sequences*i+j],18,1)
                    if dtw_d!=-1:
                        org_dtw_ds.append(dtw_d)
                    org_mfe_diffs.append((RNA.fold(test_seqs[i*num_sequences+j])[1]-RNA.fold(ref_seqs[i])[1])/len(ref_seqs[i])*3)
                    org_jacs.append(Jaccard_Coeff(test_seqs[i*num_sequences+j],ref_seqs[i]))
                    c,bp=codon_bp_similarity(test_seqs[i*num_sequences+j],ref_seqs[i])
                    org_cs.append(c)
                    org_bps.append(bp)

            dtw_ds.append(org_dtw_ds)
            mfe_diffs.append(org_mfe_diffs)
            jacs.append(org_jacs)
            cs.append(org_cs)
            bps.append(org_bps)


    # print(f'CAIs:{cais}')
    # print(f'GCs percentage:{gcs}')
    if len(ref_cais)==NUM_ORGANISMS and len(ref_gcs)==NUM_ORGANISMS:
        other_evaluations=[jacs,cs,bps,dtw_ds,mfe_diffs]
        # 中位数信息统计
        natural_csi_medians = []
        natural_gc_medians=[]
        infer_csi_medians,infer_gc_medians=[],[]
        other_eval_medians=[[] for _ in range(len(other_evaluations))]
        for i in range(NUM_ORGANISMS):
            natural_csi_medians.append(np.median(ref_cais[i]))
            natural_gc_medians.append(np.median(ref_gcs[i]))
            infer_csi_medians.append(np.median(cais[i]))
            infer_gc_medians.append(np.median(gcs[i]))
        for i in range(len(other_evaluations)):
            for j in range(NUM_ORGANISMS):
                other_eval_medians[i].append(np.median(other_evaluations[i][j]))

        # 中位数信息输出
        osize = tuple(len(rc) for rc in ref_cais)
        print(f'Model:{plmodel_path}')
        print(f'Dataset size:{osize},Deterministic:{deterministic},Temperature:{temperature},Top_p:{top_p},Sampling number:{num_sequences}\n')
        print(f'Natural CSI medians:{natural_csi_medians}')
        print(f'Inference CSI medians:{infer_csi_medians}')
        print(f'Natural GC medians:{natural_gc_medians}')
        print(f'Inference GC medians:{infer_gc_medians}')
        labels = ['Jaccard coefficient', 'Codon similarity', 'Basepair similarity', 'Dynamic time warping distance',
                  'MFE difference']
        for i in range(len(labels)):
            print(f'{labels[i]} medians: {other_eval_medians[i]}'+f'\nInterval: [{np.min(other_eval_medians[i])}'
                    +f'-{np.max(other_eval_medians[i])}]')

        # mfe_difference和CSI、GC距离刻画
        mfe_d,csi_d,gc_d=[],[],[]
        for i in range(NUM_ORGANISMS):
            mfe_d.append(np.sqrt(np.mean(np.array(mfe_diffs[i])**2)))
            ex_ref_cai,ex_ref_gc=[],[]
            if num_sequences>1:
                for j in ref_cais[i]:
                    ex_ref_cai.extend([j] * num_sequences)
                for j in ref_gcs[i]:
                    ex_ref_gc.extend([j] * num_sequences)
            else:
                ex_ref_cai, ex_ref_gc = ref_cais[i], ref_gcs[i]
            assert(len(ex_ref_cai)==len(cais[i]))
            assert(len(ex_ref_gc)==len(gcs[i]))
            csi_d.append(np.sqrt(np.mean((np.array(cais[i])-np.array(ex_ref_cai)) ** 2)))
            gc_d.append(np.sqrt(np.mean((np.array(gcs[i])-np.array(ex_ref_gc)) ** 2)))

        print(f'CSI RMSE distance:{csi_d}')
        print(f'GC RMSE distance:{gc_d}')
        print(f'MFE RMSE distance:{mfe_d}')
        print(f'Inference time elapsed:{duration:.2f}s')
        draw_comp_boxes(ref_cais,cais,ref_gcs,gcs,deterministic,temperature,top_p,num_sequences)
        # jac/codon/basepair/dtw_d/mfe （纵向）按物种箱线图绘制，标题包含超参数信息
        draw_boxes(other_evaluations,labels,deterministic,temperature,top_p,num_sequences)
        plt.show()

    return df,cais,gcs


def draw_comp_boxes(ref_cais,cais,ref_gcs,gcs,deterministic,temperature,top_p,num_sequences):
    size = tuple(len(rc) for rc in ref_cais)
    # draw CAI attribute
    caifig, axes = plt.subplots(1, NUM_ORGANISMS,figsize=(19.2,10.8))
    caifig.suptitle(f'CAI Attribute of Inference and Reference Dataset(Size={size})\n' +
                    f'Deterministic:{deterministic},Temperature:{temperature},Top_p:{top_p},Infer_number(per):{num_sequences}')
    plt.tight_layout()
    colors = ['black', 'red', 'green', 'blue', 'yellow']
    light_colors = ['grey', 'lightcoral', 'lightgreen', 'lightblue', 'lightyellow']
    for i, k in enumerate(ORGANISM2ID2.keys()):
        axes[i].set_title(f'{k}', fontdict={'color': colors[i]}, loc='center', pad=10)
        print(len(ref_cais[i]), len(cais[i]))
        # assert(len(ref_cais[i])==len(cais[i]))
        axes[i].boxplot([ref_cais[i], cais[i]], tick_labels=["REFER", "INFER"], vert=True,
                        patch_artist=True, boxprops={'facecolor': light_colors[i], 'edgecolor': colors[i],
                                                     'linewidth': 2.5, 'linestyle': '-', 'alpha': 0.8}, showfliers=True,
                        showmeans=True)

    curr_t=datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'CSI_{curr_t}.pdf',format='pdf')

    # draw GC attribute
    gcfig, gaxes = plt.subplots(1, NUM_ORGANISMS,figsize=(19.2,10.8))
    gcfig.suptitle(f'GC Attribute of Inference and Reference Dataset(Size={size})\n' +
                   f'Deterministic:{deterministic},Temperature:{temperature},Top_p:{top_p},Infer_number(per):{num_sequences}')
    plt.tight_layout()
    for i, k in enumerate(ORGANISM2ID2.keys()):
        gaxes[i].set_title(f'{k}', fontdict={'color': colors[i]}, loc='center', pad=10)
        # print(len(ref_gcs[i]), len(gcs[i]))
        # assert(len(ref_cais[i])==len(cais[i]))
        gaxes[i].boxplot([ref_gcs[i], gcs[i]], tick_labels=["REFER", "INFER"], vert=True,
                         patch_artist=True, boxprops={'facecolor': light_colors[i], 'edgecolor': colors[i],
                                                      'linewidth': 2.5, 'linestyle': '-', 'alpha': 0.8},
                         showfliers=True, showmeans=True)

    curr_t=datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'GC_{curr_t}.pdf',format='pdf')
    # plt.show()



def draw_boxes(other_evals,labels,deterministic,temperature,top_p,num_sequences):
    size = [len(other_evals[0][i]) // num_sequences for i in range(NUM_ORGANISMS)]
    other_evfig, other_evaxes = plt.subplots(len(other_evals), NUM_ORGANISMS, figsize=(19.2, 10.8))
    other_evfig.suptitle(
        'Jaccard coefficient/Codon similarity/Basepair similarity/Dynamic time warping distance/MFE difference' +
        f'\nDataset size:{size},Deterministic:{deterministic},Temperature:{temperature},Top_p:{top_p},Sampling number:{num_sequences}')
    plt.tight_layout()
    colors = ['black', 'red', 'green', 'blue', 'yellow']
    light_colors = ['grey', 'lightcoral', 'lightgreen', 'lightblue', 'lightyellow']
    #  labels=['Jaccard coefficient','Codon similarity','Basepair similarity','Dynamic time warping distance','MFE difference']
    for i in range(len(other_evals)):
        for j in range(NUM_ORGANISMS):
            other_evaxes[i,j].set_title(list(ORGANISM2ID2.keys())[j],fontdict={'color': colors[j]}, loc='center', pad=10)
            other_evaxes[i,j].boxplot(other_evals[i][j],tick_labels=[labels[i]],vert=True,
                         patch_artist=True, boxprops={'facecolor': light_colors[j], 'edgecolor': colors[j],
                                                      'linewidth': 2.5, 'linestyle': '-', 'alpha': 0.8},
                         showfliers=True, showmeans=True)

    curr_t=datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'Other_Evals_{curr_t}.pdf',format='pdf')
    # plt.show()


def data_extract(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError("The assigned file is not found.")
    if data_path[-5:]=='fasta':
        return prepare_data_from_fasta_for_infer(data_path)
    elif data_path[-5:]=='jsonl' or data_path[-4:]=='json':
        return IterableJSONData(data_path)
    else:
        raise ValueError(f"Unsupported data type:{data_path.split('.')[-1]}")


# 统一从json或者jsonl文件读取DNA信息
def cal_cai(json_dataset,test_sequences,organisms,genetic_code=11,weights_by_org:dict=None):
    if weights_by_org is None:
        if not os.path.isfile('Codon_Frequency_Table.xlsx'):
            if not isinstance(json_dataset, IterableJSONData):
                raise ValueError("json_dataset should be type class IterableJSONData.")
            if len(test_sequences) != len(organisms):
                raise ValueError("test_sequences should match in length with organisms.")
            if not isinstance(organisms[0], str):
                raise ValueError("organisms should only contain strings.")

            seqs_by_org = dict()
            for v in ORGANISM2ID2.values():
                seqs_by_org[v] = []
            for sa in json_dataset:
                seq = sa.get('codons').split(' ')
                seq = [codon.split(STOP_SYMBOL)[-1] for codon in seq]
                seqs_by_org[sa.get('organism')].append(''.join(seq))

            weights_by_org = dict()
            for k, v in ORGANISM2ID2.items():
                weights_by_org[k] = CAI.relative_adaptiveness(sequences=seqs_by_org[v], genetic_code=genetic_code)
            df = pd.DataFrame(weights_by_org)
            df.to_excel('Codon_Frequency_Table.xlsx')
        else:
            # 不同于三种架构的模型训练代码中的读取Excel-转化字典部分
            df=pd.read_excel('Codon_Frequency_Table.xlsx',index_col=0)
            weights_by_org={c:df[c].to_dict() for c in df.columns}

    cais = []
    for te_seq, organism in zip(test_sequences, organisms):
        if te_seq=="":
            break
        if list(te_seq).count('U') + list(te_seq).count('N') + list(te_seq).count('K') > len(te_seq) * 0.05:
            print('UNK percentage of label sequence is over 5%.')
            continue
        te_seq = te_seq.replace('UNK', '')
        cais.append(CAI.CAI(te_seq, weights=weights_by_org[organism], genetic_code=genetic_code))

    # print(f'CAIs:{tuple(zip(organisms, cais))}')
    print(f'CAI-mean and std:{(np.mean(cais),np.std(cais))}')

    return cais,weights_by_org


def cal_gc(test_sequences,organisms):
    gcs=[]
    for seq,org in zip(test_sequences,organisms):
        if seq=="":
            break
        if list(seq).count('U')+list(seq).count('N')+list(seq).count('K')>len(seq)*0.05:
            print('UNK percentage of label sequence is over 5%.')
            continue
        seq=seq.replace('UNK','')
        gc_count=list(seq).count('G')+list(seq).count('C')
        gcs.append(gc_count/len(seq)*100)

    # print(f'GCs:{tuple(zip(organisms,gcs))}')
    print(f'GC-mean and std:{(np.mean(gcs),np.std(gcs))}\n')
    return gcs



if __name__=="__main__":
    # 将 NumPy 的基础重建函数添加到安全全局变量列表
    # torch.serialization.add_safe_globals([__main__.MyBigBirdModel])
    plmodel_path='model/dataset_re_stru_late_0.3_0.01_32.ckpt'
    data_path='dataset/Tests.xlsx'
    jdata_path='dataset/valid_data_0.1_0.3_0.8.jsonl'
    # inference_from_str(plmodel_path,data_path)
    # fasta_json_ds=data_extract(data_path)
    dataset=IterableJSONData(jdata_path)
    inference(plmodel_path,dataset,size=30,deterministic=True,temperature=0.2,top_p=0.95,num_sequences=1,privil=True)



