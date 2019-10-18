def idn_features(df, col_name='idn', curr_year=2019, drop=True):
        loc_map = {110000: '北京市',120000: '天津市',130000: '河北省',140000: '山西省',150000: '内蒙古自治区',210000: '辽宁省',
                   220000: '吉林省',230000: '黑龙江省',310000: '上海市',320000: '江苏省',330000: '浙江省',340000: '安徽省',
                   350000: '福建省',360000: '江西省',370000: '山东省',410000: '河南省',420000: '湖北省',430000: '湖南省',
                   440000: '广东省',450000: '广西壮族自治区',460000: '海南省',500000: '重庆市',510000: '四川省',520000: '贵州省',
                   530000: '云南省',540000: '西藏自治区',610000: '陕西省',620000: '甘肃省',630000: '青海省',640000: '宁夏回族自治区',
                   650000: '新疆维吾尔自治区',710000: '台湾省',810000: '香港特别行政区',820000: '澳门特别行政区'}
        df['age'] = df[col_name].apply(lambda x: curr_year-int(x[6:10]))
        df['gender']=df[col_name].apply(lambda x: 1 if (x[-1]=='X' or x[-1]=='x' or int(x[-1])%2!=0) else 0) # 1: F 0: M
        
        # household_province = df[col_name].apply(lambda x: loc_map[int(x[0:2]+'0000')])
        # df['household_province_gdp_rank']=household_province.apply(FeatureExtractor.province_gdp_rank)
        # df['household_province_population_rank']=household_province.apply(FeatureExtractor.province_population_rank)
        # df['household_province_edu_rank']=household_province.apply(FeatureExtractor.province_edu_rank)
        
        if drop:
            df.drop([col_name],1,inplace=True)
        return df


    
def province_gdp_rank(province):
        if province in ['北京市','上海市','天津市','江苏省','浙江省','福建省','广东省','香港特别行政区','澳门特别行政区']:
            return 3
        elif province in ['山东省','内蒙古自治区','湖北省','重庆市','陕西省','辽宁省','吉林省']:
            return 2
        elif province in ['宁夏回族自治区','湖南省','海南省','河南省','新疆维吾尔自治区','四川省','安徽省','河北省','青海省','江西省']:
            return 1
        elif province in ['山西省','黑龙江省','西藏自治区','广西壮族自治区','贵州省','云南省','甘肃省']:
            return 0
        else:
            return 0
        

def province_population_rank(province):
        if province in ['广东省','山东省','河南省','四川省','江苏省','河北省','湖南省','安徽省']:
            return 3
        elif province in ['湖北省','浙江省','广西壮族自治区','云南省','江西省','辽宁省','黑龙江省','陕西省']:
            return 2
        elif province in ['山西省','福建省','贵州省','重庆市','甘肃省','内蒙古自治区','上海市','新疆维吾尔自治区']:
            return 1
        elif province in ['北京市','天津市','海南省','香港特别行政区','宁夏回族自治区','青海省','西藏自治区', '澳门特别行政区']:
            return 0
        else:
            return 0
        
# refer http://www.ccutu.com/208510.html
def province_edu_rank(province):
        if province in ['北京市','江苏省','上海市','湖北省','山东省','广东省','浙江省','辽宁省','陕西省','香港特别行政区']:
            return 3
        elif province in ['四川省','河南省','湖南省','黑龙江省','安徽省','福建省','吉林省','河北省']:
            return 2
        elif province in ['天津市','重庆市','江西省','广西壮族自治区','山西省','云南省','甘肃省','贵州省','澳门特别行政区']:
            return 1
        elif province in ['内蒙古自治区','新疆维吾尔自治区','海南省','宁夏回族自治区','青海省','西藏自治区']:
            return 0
        else:
            return 0