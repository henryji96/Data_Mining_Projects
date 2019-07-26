import pandas as pd
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)




import numpy as np
#显示所有
np.set_printoptions(threshold=np.inf)
#显示更多列
np.set_printoptions(linewidth=100)
