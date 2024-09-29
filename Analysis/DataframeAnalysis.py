import os
import numpy as np
import pyarrow.parquet as pq
from arch.unitroot import ADF, PhillipsPerron, DFGLS, KPSS, ZivotAndrews, VarianceRatio
from scipy.fftpack import fft, fftfreq
import torch
import pandas as pd

class DataframeAnalysis():
    def __init__(self, root_path,data_path):
        self.root_path = root_path
        self.data_path = data_path
        if data_path.endswith('.csv'):
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            self.df_raw = df_raw
        elif data_path.endswith('.xlsx'):
            df_raw = pd.read_excel(os.path.join(self.root_path, self.data_path))
            self.df_raw = df_raw
        elif data_path.endswith('.parquet'):
            parquet_file = pq.ParquetFile(os.path.join(self.root_path, self.data_path))
            df_raw = parquet_file.read().to_pandas()
            self.df_raw = df_raw
    
    #* 统计量
    def getShape(self):
        # 获取数据形状：（序列长度，变量数）
        return self.df_raw.shape
    
    def getAverageColumn(self,start_col=None, end_col=None):
        # 获取数据每一列的均值
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        average = df.mean(axis=0)
        average_df = pd.DataFrame()
        average_df['feature'] = average.index
        average_df['average'] = average.values
        return average_df

    
    def getVarianceColumn(self, start_col=None, end_col=None):
        # 获取数据每一列的方差
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        var = df.var(axis=0)
        var_df = pd.DataFrame()
        var_df['feature'] = var.index
        var_df['variance'] = var.values
        return var_df
    
    def getStdColumn(self, start_col=None, end_col=None):
        # 获取数据每一列的标准差
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        std = df.std(axis=0)
        std_df = pd.DataFrame()
        std_df['feature'] = std.index
        std_df['standard deviation'] = std.values
        return std_df
    
    def getMedianColumn(self, start_col=None, end_col=None):
        # 获取数据每一列的中位数
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        median = df.median(axis=0)
        median_df = pd.DataFrame()
        median_df['feature'] = median.index
        median_df['median'] = median.values
        return median_df
    
    def getQuantileColumn(self,percent=[1/4,2/4,3/4], start_col=None, end_col=None):
        # 获取数据每一列的分位数：定义percent值以设置分为数
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        quantile = df.quantile(percent,axis=0)
        return quantile
    
    def getMaxColumn(self, start_col=None, end_col=None):
        # 获取数据每一列的最大值
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        maxval = df.max(axis=0)
        maxval_df = pd.DataFrame()
        maxval_df['feature'] = maxval.index
        maxval_df['max value'] = maxval.values
        return maxval_df
    
    def getMinColumn(self,start_col=None, end_col=None):
        # 获取数据每一列的最小值
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        minval = df.min(axis=0)
        minval_df = pd.DataFrame()
        minval_df['feature'] = minval.index
        minval_df['min value'] = minval.values
        return minval_df
    
    #* 相关性
    #todo: 自相关性与互相惯性指定序列计算接口
    def getCorr(self, method='pearson', start_col=None, end_col=None):
        # 获取所有序列两两之间的互相关性：定义method以指定计算相关性标准（'pearson' | 'kendall' | 'spearman'）
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        return df.corr(method)
    
    def getSelfCorr(self,lag=1, start_col=None, end_col=None):
        # 获取所有序列自相关系数：定义lag以指定计算自相关的滞后期数（时间间隔）
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        autocorr_lag_dict = {}
        for i in df.columns:
            ts = pd.Series(self.df_raw[i].values, index=self.df_raw.index)
            autocorr_lag = ts.autocorr(lag)
            autocorr_lag_dict[i] = autocorr_lag
        autocorr_df = pd.DataFrame(autocorr_lag_dict.items(), columns=['feature', 'self correlation'])
        return autocorr_df

    #* 平稳性
    def getADF(self, start_col=None, end_col=None):
        # 获取所有序列的ADF平稳性测试结果
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        ADFresult = {}
        for i in df.columns:
            result = ADF(self.df_raw[i].values)
            ADFresult[i] = {"Test Statistic":result.stat, "P-value":result.pvalue, "Lags":result.lags, "Trend":result.trend, "Summary":result.summary()}
        return ADFresult
    
    def getPhillipsPerron(self, start_col=None, end_col=None):
        # 获取所有序列的Phillips-Perron平稳性测试结果
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        PhillipsPerronresult = {}
        for i in df.columns:
            result = PhillipsPerron(self.df_raw[i].values)
            PhillipsPerronresult[i] = {"Test Statistic":result.stat, "P-value":result.pvalue, "Lags":result.lags, "Trend":result.trend, "Summary":result.summary()}
        return PhillipsPerronresult
    
    def getDFGLS(self, start_col=None, end_col=None):
        # 获取所有序列的DF-GLS平稳性测试结果
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        DFGLSresult = {}
        for i in df.columns:
            result = DFGLS(self.df_raw[i].values)
            DFGLSresult[i] = {"Test Statistic":result.stat, "P-value":result.pvalue, "Lags":result.lags, "Trend":result.trend, "Summary":result.summary()}
        return DFGLSresult

    def getKPSS(self, start_col=None, end_col=None):
        # 获取所有序列的KPSS平稳性测试结果
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        KPSSresult = {}
        for i in df.columns:
            result = KPSS(self.df_raw[i].values)
            KPSSresult[i] = {"Test Statistic":result.stat, "P-value":result.pvalue, "Lags":result.lags, "Trend":result.trend, "Summary":result.summary()}
        return KPSSresult
    
    def getZivotAndrews(self, start_col=None, end_col=None):
        # 获取所有序列的Zivot-Andrew平稳性测试结果
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        ZivotAndrewsresult = {}
        for i in df.columns:
            result = ZivotAndrews(self.df_raw[i].values)
            ZivotAndrewsresult[i] = {"Test Statistic":result.stat, "P-value":result.pvalue, "Lags":result.lags, "Trend":result.trend, "Summary":result.summary()}
        return ZivotAndrewsresult
    
    def getVarianceRatio(self, start_col=None, end_col=None):
        # 获取所有序列的Variance Ratio平稳性测试结果
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        df = self.df_raw.loc[:,start_col:end_col]
        VarianceRatioresult = {}
        for i in df.columns:
            result = VarianceRatio(self.df_raw[i].values)
            VarianceRatioresult[i] = {"Test Statistic":result.stat, "P-value":result.pvalue, "Lags":result.lags, "Trend":result.trend, "Summary":result.summary()}
        return VarianceRatioresult
    
    #* 周期性分析
    #todo 具体实现仍然需要讨论，先了解一下现有主流方法
    def getFFTtopk(self,col,top_k_seasons=3):
        # 获得k个最主要的周期
        if col not in self.df_raw.columns:
            print(f"column {col} not found")
            return None
        fft_series = fft(self.df_raw.loc[:, col].values)
        power = np.abs(fft_series)
        sample_freq = fftfreq(fft_series.size)
        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        powers = power[pos_mask]
        top_k_ids = np.argpartition(powers, -top_k_seasons)[-top_k_seasons:]
        top_k_power = powers[top_k_ids]
        fft_periods = (1 / freqs[top_k_ids]).astype(int)
        sample_freq = pd.DataFrame(sample_freq, columns=['fft results'])
        return {"top_k_power": top_k_power, "fft_periods": fft_periods}, sample_freq

    #* 缺失值分析
    def getNanIndex(self,start_col=None, end_col=None):
        # 获得包含缺失值的index条目
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        data_nan_time = self.df_raw[self.df_raw.loc[:, start_col:end_col].isnull().values==True].index.unique()
        return data_nan_time
    
    def getInterpolate(self, start_col=None, end_col=None, **kwargs):
        # 插值填补函数(通过**kwargs传入interpolate函数的参数)
        if start_col==None:
            start_col = self.df_raw.columns[0]
        if end_col==None:
            end_col = self.df_raw.columns[-1]
        NullNum = self.df_raw.loc[:,start_col:end_col].isnull().sum()
        if True in [i>0 for i in NullNum]:
            print('kwargs:' , kwargs)
            new_df = self.df_raw.loc[:, start_col:end_col].interpolate(**kwargs)
        else:
            new_df = self.df_raw.loc[:, start_col:end_col]
        self.df_raw.loc[:, start_col:end_col] = new_df
        return self.df_raw

