import diskcache
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path

from pandarallel import pandarallel
from GEOparse import GEOparse

pandarallel.initialize(verbose=False)


def read_df_path(df_path):
    if isinstance(df_path, pd.DataFrame):
        df = df_path.copy()
    else:
        df_path = Path(df_path)
        df = pd.read_csv(df_path, low_memory=False)
    return df


class MethyPretrainDataset(Dataset):
    def __init__(self, df_path, data_root, df_map_root,
                 col_path='path',
                 num_classes=20, mask_ratio=0.15,
                 max_seq_len=None,
                 config=None,
                 training=True,
                 cache_dir=None,
                 ):
        """
        - col_seq: tokens
        - col_label: labels
        """

        self.col_path = col_path
        self.training = training
        self.num_classes = num_classes
        self.num_methy_bin = self.num_classes
        self.mask_ratio = mask_ratio
        self.max_seq_len = max_seq_len

        self.df = read_df_path(df_path)
        self.data_root = Path(data_root)

        self.df_map_root = Path(df_map_root)
        self.methy_maps = {}
        for path in self.df_map_root.glob('*.std.csv'):
            name = str(path.name).split('.')[0]
            df_map = pd.read_csv(path, index_col=0, low_memory=False)
            df_map['pos_id'] = (df_map['pos'] // 5000).clip(upper=50000)
            df_map = df_map.drop(columns=['pos'])
            df_map['chr_id'] = df_map['chr'].replace({'X': '23', 'Y': '24'}).astype('int')
            df_map = df_map.drop(columns=['chr'])
            self.methy_maps[name] = df_map

        self.cache_dir = cache_dir
        if self.cache_dir is None:
            self.diskcache = None
        else:
            self.diskcache = diskcache.Cache(directory=self.cache_dir, eviction_policy='none')

    def __len__(self):
        return len(self.df)

    def get_sample_cache(self, key, cache):
        if cache is None:
            values = self.get_sample(*key)
            return values

        if key in cache:
            values = cache[key]
            return values

        values = self.get_sample(*key)
        cache[key] = values
        return values

    def get_sample(self, group, methy_map, path, chr_id):
        # load data
        if group == 'tcga':
            df_case = pd.read_csv(path, sep='\t', header=None)
            df_case.columns = ['site', 'methy']
        elif group == 'geo':
            result = GEOparse.get_GEO(filepath=str(path), how='brief', silent=True)
            df_case = result.table
            df_case = df_case.rename(columns={'ID_REF': 'site', 'VALUE': 'methy'})
            df_case = df_case[['site', 'methy']]
        else: # other
            df_case = pd.read_csv(path)
            df_case = df_case.rename(columns={'Site': 'site', 'Frac': 'methy','Freq':'freq'})

        df_case = df_case.merge(self.methy_maps[methy_map], left_on='site', right_index=True, how='inner')
        df_case = df_case.sort_values(['chr_id', 'pos_id'])
        df_case = df_case[df_case['chr_id'] != 24]  # remove Y chromosome
        df_case['methy'] = df_case['methy'].astype('float32').fillna(-1)
        df_case['freq'] = np.where(df_case['methy'].isna(),0,df_case.get('freq',20))
        df_case['freq'] = df_case['freq'].fillna(20)
        df_case = df_case.drop(columns=['site', 'gene'])
        return df_case

    def _get_chr(self, df_case, sampling=True):
        methy_mean = df_case['methy'].mean()
        freq_mean = df_case['freq'].mean()
        if np.isnan(freq_mean):
            freq_mean = 0

        # sampling
        if (self.max_seq_len is not None) and sampling and (len(df_case) > 0):
            ratio = max(0.1, np.random.rand())  # uniform sampling
            n = min(len(df_case), int(self.max_seq_len * ratio))
            sample_p = np.random.rand()
            if sample_p < 0.5:  # random sampling
                df_case = df_case.sample(n=n)
            else:  # sequential sampling
                s = np.random.randint(0, len(df_case))
                df_case = df_case.iloc[s:s + n]

        # methylation token: 0[PAD] 1[NA] 2-21[0-1]
        if len(df_case) > 0:
            methy = df_case['methy'].values
            freq = df_case['freq'].values
            chromo = df_case['chr_id'].values
            pos = df_case['pos_id'].values + 1
            gene = df_case['gene_id'].values + 3
        else:  # 填0
            methy = np.array([-2, -2], dtype='float32')
            freq = np.array([0, 0], dtype='int')
            chromo = np.array([0, 0], dtype='int')
            pos = np.array([0, 0], dtype='int')
            gene = np.array([0, 0], dtype='int')
            methy_mean = 0.

        inputs = {
            'methy': methy,
            'freq': freq,
            'chromo': chromo,  # padding 0
            'pos': pos,  # padding 0
            'gene': gene,  # padding 0
        }
        pad_values = {
            'methy': -2.,
        }
        if self.max_seq_len is not None:
            pad_length = self.max_seq_len - len(methy)
            if pad_length <= 0:
                for k in list(inputs.keys()):
                    inputs[k] = inputs[k][:self.max_seq_len]
            else:
                for k in list(inputs.keys()):
                    pad_value = pad_values.get(k, 0)
                    inputs[k] = np.pad(inputs[k], [(0, pad_length)], mode='constant', constant_values=pad_value)

        # chromosome token
        inputs = {
            'methy': np.concatenate([np.array([methy_mean]), inputs['methy'][:-1]]).astype('float32'),
            'freq': np.concatenate([np.array([freq_mean]), inputs['freq'][:-1]]),
            'chromo': np.concatenate([chromo[:1], inputs['chromo'][:-1]]),
            'pos': np.concatenate([np.array([0], dtype='int'), inputs['pos'][:-1]]),
            'gene': np.concatenate([np.array([1], dtype='int'), inputs['gene'][:-1]]),
        }
        return inputs

    def __getitem__(self, idx):
        case = self.df.iloc[idx]
        path = self.data_root / case[self.col_path]
        group = case['group']
        methy_map = case['methy_map']

        key = (group, methy_map, path, None)
        df_case = self.get_sample_cache(key, self.diskcache)

        # select one chromosome
        chr_id = df_case['chr_id'].sample().item()
        df_case_chr = df_case[df_case['chr_id'] == chr_id]
        inputs = self._get_chr(df_case_chr, sampling=True)

        # masking
        labels = pd.cut(pd.Series(inputs['methy']),
                        bins=[-np.inf,-1] + list(np.linspace(0, 1, self.num_methy_bin + 1))[:-1] + [np.inf],
                        right=False).cat.codes.values.astype('int')
        chr_token_methy = labels[0]
        labels[labels <= 1] = -100
        p = np.random.rand(len(labels))
        labels[p <= 0.85] = -100
        inputs['methy'][labels != -100] = -1
        inputs['freq'][labels != -100] = 0
        labels[0] = chr_token_methy
        inputs['methy'][0] = -1
        inputs['freq'][0] = 0

        return inputs, labels


class MethyDiagDataset(MethyPretrainDataset):
    def __init__(self, *args, freq_filter=None, **kargs):
        super(MethyDiagDataset, self).__init__(*args, **kargs)
        self.col_label = 'label_cancer'
        self.df[self.col_label] = self.df[self.col_label].fillna(-100)
        self.labels = self.df[self.col_label].astype('int').values
        self.num_classes = 1
        self.freq_filter = freq_filter


    def __getitem__(self, idx):
        case = self.df.iloc[idx]
        path = self.data_root / case[self.col_path]
        group = case['group']
        methy_map = case['methy_map']
        label = self.labels[idx]

        key = (group, methy_map, path, None)
        df_case = self.get_sample_cache(key, self.diskcache)
        inputs_list = []
        for chr_id in range(1, 24):
            df_case_chr = df_case[df_case['chr_id'] == chr_id]
            if self.freq_filter is not None:
                df_case_chr = df_case_chr[df_case_chr['freq']>=self.freq_filter]
            inputs_chr = self._get_chr(df_case_chr, sampling=False)
            inputs_list += [inputs_chr]
        keys = list(inputs_list[0].keys())
        inputs = {}
        for key in keys:
            val = np.stack([x[key] for x in inputs_list])
            inputs[key] = val
        return inputs, np.array([label], dtype='int')

