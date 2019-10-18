class EDA(object):

    @staticmethod
    def feature_class_count(df, feature):
        """
        Unique class counts for one feature

        df: DataFrame
        feature: str
        """
        print(df[feature].value_counts())

    @staticmethod
    def num_unique_class(df, column_list):
        """
        Print # of unique class for list of discrete features


        df: DataFrame
        column_list: list of feature name
        """
        for i, col_name in enumerate(column_list):
            print("{:>2}: {:>2} {}".format(i, df[col_name].nunique(), col_name))

    @staticmethod
    def correlation_matrix(df):
      f = plt.figure(figsize=(20, 20))
      plt.matshow(df.corr(), fignum=f.number)
      plt.xticks(range(df.shape[1]), df.columns, fontsize=12, rotation=90)
      plt.yticks(range(df.shape[1]), df.columns, fontsize=12)
      cb = plt.colorbar()
      cb.ax.tick_params(labelsize=8)

      plt.show()

    @staticmethod
    def get_top_abs_correlations(df, n=10):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))

        au_corr = df.corr().abs().unstack()
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]
