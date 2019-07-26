import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import axes3d
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

#import sys
#sys.path.append('../utils')

class VisualizeTools(object):
    
    # extract age, gender, household_province information from id card
    # http://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap
    @staticmethod
    def corr_heatmap(df, figsize=(10, 10), dpi=400, save_path=None, show=True):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

        plt.figure(figsize=figsize, dpi=dpi )
        # camp = Set1, Set2, YlGnBu, Other Color palettes
        ax = sns.heatmap(df.corr().applymap(lambda x: round(x, 3)), annot=True, linewidths=.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()
            
    @staticmethod
    def corr_with_target(df, target_name):
        return pd.DataFrame(df.corr()[target_name].sort_values(ascending=False))
    
    @staticmethod
    def scatter_3D(df, target):
        '''
            df: 前3列为xyz
            target： y值用于绘制颜色； 大于2类时需修改colors_map, legend
        '''
        x, y, z=df[:,1],df[:,2],df[:,3]

        # blue, cyan, green, black, magenta, red, white, yellow
        colors_map = {0: 'c', 1: 'red'}
        colors = target.map(colors_map)

        # Create plot
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(1, 1, 1)
        ax = fig.gca(projection='3d')

        for x, y, z, color in zip(x, y, z, colors):
            ax.scatter(x, y, z, alpha=0.8, c=color, edgecolors='none', s=30)

        ax.set_xlabel('X axis')  
        ax.set_ylabel('Y axis')    
        ax.set_zlabel('Z axis') 
        ax.set_title('3D Scatter Plot')  

        # plt.legend(loc=2)
        plt.legend('10') # target 1 red; 0 cyan 
        plt.show()
        
    # http://seaborn.pydata.org/generated/seaborn.PairGrid.html#seaborn.PairGrid
    @staticmethod
    def scatter_matrix1(df, hue_col, figsize=(10, 10), dpi=600, save_path=None, show=False):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        
        plt.figure(figsize=figsize, dpi=dpi)
        g = sns.PairGrid(df, hue=hue_col, palette="Set2") # hue_kws={"marker": ["o", "s", ""D""]} 
        g = g.map(plt.scatter, linewidths=1, edgecolor="w", s=40)
        g = g.add_legend()

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()
            
    @staticmethod
    def scatter_matrix2(df, hue_col, figsize=(10, 10), save_path=None, dpi=600):
        '''
        scatter matrix + kdeplot
        '''
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        
        plt.figure(figsize=figsize,dpi=dpi)
        g = sns.PairGrid(df)
        g = g.map_upper(plt.scatter)
        g = g.map_lower(sns.kdeplot, cmap="Blues_d")
        g = g.map_diag(sns.kdeplot, lw=3, legend=False)
        plt.savefig(save_path)

    @staticmethod
    def numeric_features_violin(df, target_col, figsize, dpi, save_path=None, show=True):  
        features = df.columns.drop(target_col)
        num_of_plots = len(features)
        target_unique_val = np.unique(df[target_col])

        plt.figure(figsize=figsize, dpi=dpi)
        for i in range(1, num_of_plots+1):
            plt.subplot(num_of_plots//2+1, 2, i)

            sns.violinplot(x = df[target_col], y = df[features[i-1]])

            plt.xlabel(features[i-1])
        if save_path:
            plt.savefig(save_path)          
        if show:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()
            
    @staticmethod        
    def numeric_features_kde(df, target_col, figsize, dpi, save_path=None, show=True):

        features = df.columns.drop(target_col)
        num_of_plots = len(features)
        target_unique_val = np.unique(df[target_col])

        plt.figure(figsize=figsize, dpi=dpi)
        for i in range(1, num_of_plots+1):
            plt.subplot(num_of_plots//2+1, 2, i)

            for t in target_unique_val:
                sns.kdeplot(df[df[target_col] == t][features[i-1]], label = t)
            plt.xlabel(features[i-1])
        if save_path:
            plt.savefig(save_path)          
        if show:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()      
            
    def categorical_features_count(df, target_col, figsize, dpi, save_path=None, show=True):

        features = df.columns.drop(target_col)
        num_of_plots = len(features)
        target_unique_val = np.unique(df[target_col])

        plt.figure(figsize=figsize, dpi=dpi)
        for i in range(1, num_of_plots+1):
            plt.subplot(num_of_plots//2+1, 2, i)

            sns.countplot(x = features[i-1], hue=target_col, data=df)

            plt.xlabel(features[i-1])
        if save_path:
            plt.savefig(save_path)         
        if show:
            plt.show()        
        else:
            plt.clf()
            plt.cla()
            plt.close()
        
        
        
        