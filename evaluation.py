from typing import Optional, Tuple, List
import numpy as np
import rdata
import os
import pandas as pd
from fastdtw import fastdtw


name_correction: dict[str, str] = {
    'human': "Homo sapiens (Human)",
    'mouse': "Mus musculus (Mouse)",
    "ecoli": "Escherichia coli",
    "saccharomyces": "Saccharomyces cerevisiae",
    "pichia": "Pichia angusta",
}


def generate_privileged_CFT(directory: str):
    if not os.path.isdir(directory):
        raise ValueError("Directory not found.")
    eligible_files = []
    for fn in os.listdir(directory):
        if fn.split('.')[-1] == 'rda':
            eligible_files.append(os.path.join(directory, fn))
    print(f'Detected R data files:{len(eligible_files)}\n{eligible_files}')
    weights_by_org = {}
    for ef in eligible_files:
        data_dict = rdata.read_rda(ef)
        # print(data_dict.items())
        codon_table = list(data_dict.values())[0]
        codons, aas, freqs = codon_table['V1'].tolist(), codon_table['V2'].tolist(), codon_table['V3'].tolist()
        freq_store = dict(zip(codons, freqs))
        maxf_store = dict()
        for c, a in zip(codons, aas):
            maxf_store[c] = max(codon_table[codon_table['V2'] == a]['V3'])
        for codon in freq_store.keys():
            freq_store[codon] /= maxf_store[codon]
        weights_by_org[name_correction[ef.split('_')[-1].split('.')[0]]] = freq_store

    df = pd.DataFrame(weights_by_org)
    df.to_excel('Privileged_Codon_Frequency_Table.xlsx')

    return weights_by_org


# 长度最好一致
# TODO:MinMax绘图
def dtw_distance(CFT: dict, organism: str, seq1: str, seq2: str, wsize=18, step=1) -> float:
    assert (len(seq1) == len(seq2))
    seq1 = np.array([seq1[i:i + 3] for i in range(0, len(seq1), 3)])
    seq2 = np.array([seq2[i:i + 3] for i in range(0, len(seq2), 3)])
    assert (wsize + step < len(seq1))
    if CFT.get(organism) is None:
        print('Organism codon usage data not existent.')
        return -1
    if 'UNK' in seq1 or 'UNK' in seq2:
        print('Reference sequence contains unk label.')
        return -1

    """
    mms1,mms2=[],[]
    # +1是正确设定，-1是对于Bio.Data.CodonTable缺少三个终止密码子统计的保障，对于privil=True如果rdata文件有统计终止密码子的
    # RSCU则不影响，但是为了指标评估的统一起见，忽略掉DNA序列最后的终止密码子
    for i in range(0,len(seq1)-wsize+1-1,step):
        freqs=np.array(list(map(CFT[organism].__getitem__,seq1[i:i+wsize])))
        mms1.append((np.max(freqs)-np.min(freqs))/(np.max(freqs)+np.min(freqs)))
        freqs=np.array(list(map(CFT[organism].__getitem__,seq2[i:i+wsize])))
        mms2.append((np.max(freqs)-np.min(freqs))/(np.max(freqs)+np.min(freqs)))

    mms1,mms2=np.array(mms1),np.array(mms2)
    """

    # GPU加速版本 思路：先转成频率numpy.ndarray数组，然后使用列表推导式和二维矩阵加速计算
    seq1 = np.array(list(map(CFT[organism].__getitem__, seq1[:-1])))
    seq2 = np.array(list(map(CFT[organism].__getitem__, seq2[:-1])))
    freqs = np.array([seq1[i:i + wsize] for i in range(0, len(seq1) - wsize + 1, step)])
    mms1 = ((np.max(freqs, axis=1, keepdims=False) - np.min(freqs, axis=1, keepdims=False)) /
            (np.max(freqs, axis=1, keepdims=False) + np.min(freqs, axis=1, keepdims=False)))
    freqs = np.array([seq2[i:i + wsize] for i in range(0, len(seq2) - wsize + 1, step)])
    mms2 = ((np.max(freqs, axis=1, keepdims=False) - np.min(freqs, axis=1, keepdims=False)) /
            (np.max(freqs, axis=1, keepdims=False) + np.min(freqs, axis=1, keepdims=False)))

    # dtw_d=np.sqrt(np.mean((mms1-mms2)**2))
    # align = dtw(mms1, mms2)
    # dtw_d = align.distance / len(mms1)
    dtw_d,_=fastdtw(mms1,mms2)
    dtw_d/=len(mms1)
    print(f'DTW distance:{dtw_d:.4f}')

    return dtw_d


# 不一定要求序列长度相等
def Jaccard_Coeff(seq1: str, seq2: str) -> float:
    seq1 = np.array([seq1[i:i + 3] for i in range(0, len(seq1), 3)])
    seq2 = np.array([seq2[i:i + 3] for i in range(0, len(seq2), 3)])
    assert (('UNK' not in seq1) and ('UNK' not in seq2))
    seta = set(list(seq1))
    setb = set(list(seq2))
    jaccard = len(seta & setb) / len(seta | setb)
    return jaccard


# 长度一致
def codon_bp_similarity(seq1: str, seq2: str) -> Tuple[float, float]:
    assert (len(seq1) == len(seq2))
    seqa = np.array([seq1[i:i + 3] for i in range(0, len(seq1), 3)])
    seqb = np.array([seq2[i:i + 3] for i in range(0, len(seq2), 3)])
    res = (seqa == seqb)
    codon_acc = sum(res) / len(seqa)
    # seqa=''.join(list(seq1))
    # seqb=''.join(list(seq2))
    bp_acc = 0
    for ai, bi in zip(seq1, seq2):
        bp_acc += float(ai == bi)
    bp_acc /= len(seq1)

    return codon_acc, bp_acc


if __name__ == '__main__':
    PCFT = generate_privileged_CFT('dataset')
