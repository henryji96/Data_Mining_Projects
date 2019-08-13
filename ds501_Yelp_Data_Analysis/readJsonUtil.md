

##### Import json dataset using python

```python
import json

# The business data
with open('yelp_academic_dataset_business.json') as f:
    business_data = []
    for line in f:
        business_data.append(json.loads(line))

# The checkin data
with open('yelp_academic_dataset_checkin.json') as f:
    checkin_data = []
    for line in f:
        checkin_data.append(json.loads(line))


        
# Simple check that all data was loaded
print(len(business_data))
print(len(checkin_data))
```

```
188593
157075
```



##### view the first dictionary (first row in dataframe)

It's a dictionary with key-value pairs.

```python
business_data[0]
```

```
{'business_id': 'Apn5Q_b6Nz61Tq4XzPdf9A',
 'name': 'Minhas Micro Brewery',
 'neighborhood': '',
 'address': '1314 44 Avenue NE',
 'city': 'Calgary',
 'state': 'AB',
 'postal_code': 'T2E 6L6',
 'latitude': 51.0918130155,
 'longitude': -114.031674872,
 'stars': 4.0,
 'review_count': 24,
 'is_open': 1,
 'attributes': {'BikeParking': 'False',
  'BusinessAcceptsCreditCards': 'True',
  'BusinessParking': "{'garage': False, 'street': True, 'validated': False, 'lot': False, 'valet': False}",
  'GoodForKids': 'True',
  'HasTV': 'True',
  'NoiseLevel': 'average',
  'OutdoorSeating': 'False',
  'RestaurantsAttire': 'casual',
  'RestaurantsDelivery': 'False',
  'RestaurantsGoodForGroups': 'True',
  'RestaurantsPriceRange2': '2',
  'RestaurantsReservations': 'True',
  'RestaurantsTakeOut': 'True'},
 'categories': 'Tours, Breweries, Pizza, Restaurants, Food, Hotels & Travel',
 'hours': {'Monday': '8:30-17:0',
  'Tuesday': '11:0-21:0',
  'Wednesday': '11:0-21:0',
  'Thursday': '11:0-21:0',
  'Friday': '11:0-21:0',
  'Saturday': '11:0-21:0'}}
```

Some key's value has sub dictionary

```python
# hour key's value is a sub dictionary
business_data[0]['hours']
```

```
{'Monday': '8:30-17:0',
 'Tuesday': '11:0-21:0',
 'Wednesday': '11:0-21:0',
 'Thursday': '11:0-21:0',
 'Friday': '11:0-21:0',
 'Saturday': '11:0-21:0'}
```

```python
# attribute key's value is also a sub dictionary
# We can also find that 'BusinessParking' has another dictionary
business_data[0]['attributes']
```

```
{'BikeParking': 'False',
 'BusinessAcceptsCreditCards': 'True',
 'BusinessParking': "{'garage': False, 'street': True, 'validated': False, 'lot': False, 'valet': False}",
 'GoodForKids': 'True',
 'HasTV': 'True',
 'NoiseLevel': 'average',
 'OutdoorSeating': 'False',
 'RestaurantsAttire': 'casual',
 'RestaurantsDelivery': 'False',
 'RestaurantsGoodForGroups': 'True',
 'RestaurantsPriceRange2': '2',
 'RestaurantsReservations': 'True',
 'RestaurantsTakeOut': 'True'}
```



### How to turn json format into dataframe?

##### extract key-values into python list, each list represents one feature

###### Can modify data here to extract more features(attribute list)

```python
# extract data from keys with values that's not dictionary
business_ids = []
names = []
# can extract other feature list here

for eachRow in business_data:
    business_ids.append(eachRow['business_id'])
    names.append(eachRow['name'])
    # ..... do exploration.....
```

```python
# extract data from attribute key, its value is a sub dictionary.
attribute_BikeParking = []
attribute_RestaurantsAttire = []
# can extract other attribute list here

for eachRowAttribute in business_data:
    
    # get attribute's values whose values is a dictionary
    attribute = eachRowAttribute['attributes']
    
    # get values in attribute dictionary
    # 1. attribute BikeParking
    if (attribute != None) and ('RestaurantsDelivery' in attribute.keys()):
        attribute_BikeParking.append(attribute['RestaurantsDelivery'])
    else: 
        attribute_BikeParking.append(None)
        
    # 2. attribute RestaurantsAttire    
    if (attribute != None) and ('RestaurantsAttire' in attribute.keys()):
        attribute_RestaurantsAttire.append(attribute['RestaurantsAttire'])
    else: 
        attribute_RestaurantsAttire.append(None)
        
    # 3. get other attribute's feature we want
    # ..... do exploration.....
        
print(len(attribute_BikeParking))
print(len(attribute_RestaurantsAttire))
```

```
188593
188593
```



##### turn extracted feature list above into dataframe

###### Need to modify code here when add more features (columns)

```python
import pandas as pd
from IPython.display import display
business_data_df = pd.DataFrame(data = {
    'business_ids': business_ids,
    'names': names,
    'attribute_BikeParking': attribute_BikeParking,
    'attribute_RestaurantsAttire': attribute_RestaurantsAttire
})
display(business_data_df.head(2))
print(business_data_df.shape)
```

![image](501md-1.png)