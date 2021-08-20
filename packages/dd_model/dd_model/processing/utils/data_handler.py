from __future__ import absolute_import

import re

import numpy as np
import pandas as pd
import nltk

from ddd_model.processing.utils.clean import process_token


class DataCleaner:
    def __init__(self):
        return

    def extract_fail_symptom_from_short(self, df, col):
        """
        :df: pd.DataFrame
        :col: fail symptom
        """
        # Transform Full Width to Half Width
        df_tmp = df[[col]].copy()

        # Extract Fail Symptoms
        df_tmp["reg_1"] = df_tmp[col].apply(self._reg_func1)
        df_tmp["reg_2"] = df_tmp[col].apply(self._reg_func1)
        df_tmp["reg_3"] = df_tmp[col].apply(self._reg_func1)

        # Pick the first not nan
        df_tmp["fail_symptom"] = df_tmp.apply(
            lambda x: self._summrize_func(x.reg_1, x.reg_2, x.reg_3), axis=1
        )
        df_tmp["fail_symptom"] = df_tmp.apply(
            lambda x: x[col] if pd.isnull(x.fail_symptom) else x.fail_symptom, axis=1
        )

        return df_tmp.fail_symptom.values

    def _strF2H(self, s):
        """Transform Full Width to Half Width
        :s: Text
        """
        rstring = ""
        for uchar in s:
            u_code = ord(uchar)
            if u_code == 12288:  # transform full spaces
                u_code = 32
            elif 65281 <= u_code <= 65374:  # transform other full characters
                u_code -= 65248
            rstring += chr(u_code)
        return rstring

    def _reg_func1(self, x):
        """[SIT][BIOS][WHL]"""
        tmp = re.findall(r"\[.*\]:*(.*)", x)
        if len(tmp) == 0:
            return np.nan
        else:
            return tmp[0]

    def _reg_func2(self, x):
        """M-BST/Tenjin&Llarga/SI:"""
        tmp = re.findall(r"\/[\w\-&\s\.]+\s*:(.*)", x)

        if len(tmp) == 0:
            return np.nan
        else:
            return tmp[0]

    def _reg_func3(self, x):
        '''M-IV/Okiwi /SI / Win10 RS5/ Lost "USB Type-C Ports"'''
        tmp = re.findall(r"\w+/ (.*)", x)

        if len(tmp) == 0:
            return np.nan
        else:
            return tmp[0]

    def _summrize_func(self, x1, x2, x3):
        """Pick the first not nan"""
        for idx in [x1, x2, x3]:
            if pd.isnull(idx):
                continue
            else:
                return idx
        return np.nan

    def word_tokenizer(self, input_text: str) -> str:
        return process_token(str(input_text).lower())

    def pos_filter(self, input_text, keep_tags=None):
        if keep_tags is None:
            keep_tags = [
                "CD",  # 數字
                "FW",  # 外來
                "JJ",
                "JJR",
                "JJS",  # 形容詞 比較集 最高級
                #'LS' list markers : 1)
                "NN",
                "NNS",  # 名詞 名詞複數
                "NNP",
                "NNPS",  # 專有名詞 複數
            ]

        if isinstance(input_text, str):
            words = nltk.word_tokenize(input_text)
        elif isinstance(input_text, list):
            words = input_text
        else:
            return None, None

        pos_tags = nltk.pos_tag(words)
        left_words, left_pos = [], []

        for word, pos in pos_tags:
            if pos in keep_tags:
                left_words.append(word)
                left_pos.append(pos)

        return left_words, left_pos

    def extract_sio_from_updates(self, updates):
        return list(set(re.findall(r"SIO(\d+)", str(updates), re.IGNORECASE)))

    def choose_first_dup_in_updates(self, obs_id, obs_id_array):
        for odx in obs_id_array:
            if str(odx) == obs_id:
                continue
            else:
                return str(odx)


class DataHandler:
    def __init__(self):
        return

    def truncate_subsys(self, df, threshold=30):
        tab_subsys = df.sub_sys.value_counts().reset_index()
        tab_subsys.columns = ["sub_sys", "cnt"]

        tab_subsys["sub_sys_new"] = tab_subsys["sub_sys"]
        tab_subsys.loc[
            tab_subsys["sub_sys_new"] == "Test Plans", "sub_sys_new"
        ] = "Test Plan"
        tab_subsys.loc[tab_subsys["cnt"] < threshold, "sub_sys_new"] = "Others"

        tab_subsys = (
            tab_subsys.groupby(["sub_sys", "sub_sys_new"])
            .agg({"cnt": sum})
            .reset_index()
            .sort_values("cnt", ascending=False)
            .reset_index(drop=True)
        )

        return tab_subsys


class DataPipeline:
    def __init__(self):
        return

    def find_dup_obs(
        self,
        df_obs: pd.Series,
        df: pd.DataFrame,
        tokenized_error: str = "short_desc",
        search_period: int = 30,
        is_open: bool = False,
    ):
        _df_obs = pd.DataFrame(df_obs).transpose()

        # filters
        df_time = self._search_time(_df_obs, df, search_period)
        if is_open:
            df_time = self._search_status(_df_obs, df_time)
        df_time_subsys = self._search_subsys(_df_obs, df_time)
        df_time_subsys = df_time_subsys[
            (df_time_subsys[tokenized_error].apply(len) != 0)
        ]

        _desired_short = _df_obs[tokenized_error].values[0]
        _possible_shorts = df_time_subsys[tokenized_error].values

        # compute similarity score
        df_score = self._compute_scores(_desired_short, _possible_shorts)
        df_score["obs_id"] = df_time_subsys.obs_id.values
        df_score = df_score.sort_values("simple_score", ascending=False)
        df_score = df_score[df_score.simple_score > 0]
        df_score["simple_score"] = df_score.simple_score.round(3)

        return list(
            df_score[["obs_id", "simple_score"]].itertuples(index=False, name=None)
        )

    def select_dup_obs(self, candidate: list, threshold: float = 0.5, n_dup: int = 10):
        result = [cdx for cdx in candidate if cdx[1] >= threshold]
        return result[:n_dup]

    def words_remover(self, word: list, remove_keywords: list):
        output = np.setdiff1d(word, remove_keywords)

        return list(np.unique(output))

    def _search_time(self, df_obs: pd.DataFrame, df: pd.DataFrame, search_period: int):
        df_out = df[
            (df["create_date"] - df_obs["create_date"].values[0]).dt.days.abs()
            <= search_period
        ].copy()

        return df_out

    def _search_subsys(self, df_obs: pd.DataFrame, df: pd.DataFrame):
        df_out = df[df["sub_sys"] == df_obs["sub_sys"].values[0]].copy()

        return df_out

    def _search_status(self, df_obs: pd.DataFrame, df: pd.DataFrame):
        df_out = df[df["status"] == "Open"].copy()

        return df_out

    def _compute_simple_score(self, repro_1: list, repro_2: list):
        # compare
        tmp_inter = np.intersect1d(repro_1, repro_2).tolist()
        tmp_union = len(set(repro_1 + repro_2))
        tmp_min = min(len(repro_1), len(repro_2))

        result = [
            tmp_inter,
            len(tmp_inter),
            tmp_union,
            tmp_min,
            len(tmp_inter) / tmp_min if tmp_min != 0 else np.nan,
        ]

        return result

    def _compute_scores(self, desired_short, possible_shorts):
        simple_score = [
            self._compute_simple_score(desired_short, ps)[-1] for ps in possible_shorts
        ]
        simple_score = [0 if np.isnan(idx) else idx for idx in simple_score]
        out = pd.DataFrame({"simple_score": simple_score})

        return out
