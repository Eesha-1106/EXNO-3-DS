## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="706" height="653" alt="image" src="https://github.com/user-attachments/assets/208c00eb-7323-442b-8012-5546bc619151" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

<img width="783" height="430" alt="image" src="https://github.com/user-attachments/assets/e4f106a1-4d7d-4bc2-8e4a-9934efc4ed6a" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="652" height="626" alt="image" src="https://github.com/user-attachments/assets/f453426d-7a2a-43e9-9217-a7efd6695847" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

<img width="623" height="555" alt="image" src="https://github.com/user-attachments/assets/a28b0597-f76a-432a-83a9-0d5097d6c316" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```

<img width="646" height="536" alt="image" src="https://github.com/user-attachments/assets/ccc0f381-654e-4ab5-8b04-f34750afe297" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="1068" height="622" alt="image" src="https://github.com/user-attachments/assets/cefaedc6-c90d-498d-8884-fc446498d1ce" />

```
!pip install scikit-learn==1.0.2
!pip install category_encoders

from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])

dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="1086" height="516" alt="image" src="https://github.com/user-attachments/assets/ab2c122b-02fe-4577-a8bb-50adf940acb2" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="941" height="555" alt="image" src="https://github.com/user-attachments/assets/a04b0a34-267a-46ab-9afe-1b2e3068a9eb" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```

<img width="1240" height="618" alt="image" src="https://github.com/user-attachments/assets/5a946b83-7d7e-4ce8-819b-ead739e044ac" />

```
df.skew()
```
<img width="513" height="370" alt="image" src="https://github.com/user-attachments/assets/5f04a626-a5fa-4608-9532-050f789dae3e" />

```
 np.log(df["Highly Positive Skew"])
```
<img width="410" height="663" alt="image" src="https://github.com/user-attachments/assets/604d4561-0570-4599-afe3-878c8debb794" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="459" height="675" alt="image" src="https://github.com/user-attachments/assets/6d62b0cc-e0ba-47d6-87de-5d55d69d7cd6" />

```
 np.sqrt(df["Highly Positive Skew"])
```

<img width="403" height="668" alt="image" src="https://github.com/user-attachments/assets/64bfa160-9133-4195-b074-836e2e6e43ab" />

```
np.square(df["Highly Positive Skew"])
```

<img width="417" height="668" alt="image" src="https://github.com/user-attachments/assets/75d2a46b-292f-411f-96d6-e15a69ad5e62" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1533" height="622" alt="image" src="https://github.com/user-attachments/assets/9d97187d-e090-4f8a-a93b-0fde1edaf3a6" />

```
df.skew()
```
<img width="500" height="346" alt="image" src="https://github.com/user-attachments/assets/787bbc60-474a-48b9-b6de-42d3a8961cb4" />

```
 df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
 df.skew()
```

<img width="503" height="395" alt="image" src="https://github.com/user-attachments/assets/a17c2326-1583-47a3-9aa5-97c9838e6ed5" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="1641" height="633" alt="image" src="https://github.com/user-attachments/assets/57e64a9f-8f21-4642-b78c-5dbe460a15b6" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="578" height="432" alt="download" src="https://github.com/user-attachments/assets/6c8beeee-e589-4ddb-89b2-0825127afc4d" />

```
sm.qqplot(<img width="565" height="432" alt="download" src="https://github.com/user-attachments/assets/8d6d9490-4129-4554-8262-e76490be2af4" />
np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="851" height="635" alt="image" src="https://github.com/user-attachments/assets/d3ad2bd4-281f-4d24-b5af-b7fdf471f7c9" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="565" height="432" alt="download" src="https://github.com/user-attachments/assets/888c9e2f-e31a-4666-a4c1-60734498c2b2" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

<img width="565" height="432" alt="download" src="https://github.com/user-attachments/assets/91822b5e-787f-4ce0-8984-45c80b239a89" />

```
dt=pd.read_csv("/content/titanic_dataset (1).csv")
dt
```

<img width="1658" height="657" alt="image" src="https://github.com/user-attachments/assets/8267ffb4-d50a-4d21-9cf6-9cfd013e45da" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

<img width="562" height="432" alt="download" src="https://github.com/user-attachments/assets/3f41c3ab-9764-46d9-a430-c82236f9c910" />


```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

<img width="565" height="432" alt="download" src="https://github.com/user-attachments/assets/1b43c46f-be85-44a0-b4da-a556dc98c680" />


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
