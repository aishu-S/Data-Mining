#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import entropy
import Feature_Extraction


# In[2]:


def entropy_calc(bins, clusters):
    size = len(clusters)
    num_clusters = 6
    _matrix = [[0 for _ in range(num_clusters)] for __ in range(num_clusters)]
    matrix = np.matrix(_matrix)
    
    for i in range(size):
        bin_num = bins[i]
        cluster_num = clusters[i]
        idx = (cluster_num, bin_num)
        matrix.itemset(idx, matrix.item(idx) + 1)
    
    row_sums = np.sum(matrix, axis=1)
    p_matrix = matrix / row_sums
    
    entropies = entropy(p_matrix)
    
    purities = np.max(matrix, axis=1) / row_sums
    
    total = np.sum(matrix)
    whole_entropy = np.sum(entropies * row_sums / total)
    whole_purity = np.sum(purities.reshape(1, 6) * row_sums /total)
    
    return whole_entropy, whole_purity


# In[3]:


# arr = [[1, 2], [3, 4]]
# maxes = np.max(arr, axis=1)
# mat = np.matrix(arr)
# # np.sum(mat, axis=0)

# maxes.shape
# aa.squeeze() / bb.squeeze()


# In[4]:


cgm_p1 = pd.read_csv("CGMData.csv")
insulin_p1 = pd.read_csv("InsulinData.csv")


# In[5]:


insulin_p1_rows = len(insulin_p1)
values_in_y = insulin_p1.dropna(axis=0, subset=['BWZ Carb Input (grams)'])
values_in_y =values_in_y[values_in_y['BWZ Carb Input (grams)'] !=0]
values_in_y['Datetime'] = pd.to_datetime(values_in_y['Date'] + ' ' + values_in_y['Time'])
# values_in_y.to_csv('C:/Users/aishu/Desktop/College/DM/assign 3/yvalues.csv')
cgm_p1['Datetime'] = pd.to_datetime(cgm_p1['Date'] + ' ' + cgm_p1['Time'])
# cgm_p1.to_csv('C:/Users/aishu/Desktop/College/DM/assign 3/cgmtrial.csv')


# In[ ]:





# In[6]:


rows_val_in_y = len(values_in_y)
start_times = []
end_times = []
y_index = range(0, rows_val_in_y)
values_in_y['y_index'] = y_index
values_in_y = values_in_y.set_index('y_index')
no_meal_times_start = []
no_meal_times_end = []
y_column_values = {}

for i in range(rows_val_in_y-1, 0, -1):
    
    end_time = values_in_y.at[i, 'Datetime'] + timedelta(hours = 2)
    
    if values_in_y.at[i-1, 'Datetime'] > end_time:
        start_times.append(values_in_y.at[i, 'Datetime'] - timedelta(minutes = 30))
        y_column_values[values_in_y.at[i, 'Datetime'] - timedelta(minutes = 30)] = values_in_y.at[i, 'BWZ Carb Input (grams)']
        end_times.append(end_time)
        et = end_time
        if values_in_y.at[i-1, 'Datetime'] > et + timedelta(hours = 2):
            no_meal_times_start.append(et)
            no_meal_times_end.append(et+timedelta(hours = 2))
            
    elif values_in_y.at[i-1, 'Datetime'] == end_time:
        start_times.append(values_in_y.at[i, 'Datetime'] + timedelta(hours = 1) + timedelta(minutes = 30))
        y_column_values[values_in_y.at[i, 'Datetime'] + timedelta(hours = 1) + timedelta(minutes = 30)] = values_in_y.at[i, 'BWZ Carb Input (grams)']
        end_times.append(values_in_y.at[i, 'Datetime']+timedelta(hours = 4))
        et = values_in_y.at[i, 'Datetime']+timedelta(hours = 4)
        if values_in_y.at[i-1, 'Datetime'] > et + timedelta(hours = 2):
            no_meal_times_start.append(et)
            no_meal_times_end.append(et+timedelta(hours = 2))
        
start_times.append(values_in_y.at[0, 'Datetime']-timedelta(minutes = 30))
y_column_values[values_in_y.at[0, 'Datetime']-timedelta(minutes = 30)] = values_in_y.at[0, 'BWZ Carb Input (grams)']
end_times.append(values_in_y.at[0, 'Datetime'] + timedelta(hours = 2))

# print(y_column_values)


# In[7]:


cgm_p1['IsMealData'] = ""


# In[8]:


y_meal_values = []
for k in range(len(start_times)):
    accp_data = []
    temp = 0
    for i in range(len(cgm_p1)):
        start_time = start_times[k]
        end_time = end_times[k]
        if cgm_p1.at[i,'Datetime'] >= start_time and cgm_p1.at[i, 'Datetime']<=end_time:
            cgm_p1.at[i, 'IsMealData'] = 1
            if pd.isna(cgm_p1.at[i, 'Sensor Glucose (mg/dL)']):
                temp+=1
            accp_data.append(cgm_p1.at[i, 'Sensor Glucose (mg/dL)'])
    if temp == 0 and len(accp_data) == 30:
        y_meal_values.append(y_column_values[start_times[k]])
        


# In[ ]:





# In[9]:


# cgm_p1.to_csv('C:/Users/aishu/Desktop/College/DM/assign 3/cgmtry.csv')


# In[10]:


y_value_max = values_in_y['BWZ Carb Input (grams)'].max()
y_value_min = values_in_y['BWZ Carb Input (grams)'].min()
No_of_bins = (y_value_max - y_value_min)/20
No_of_bins = round(No_of_bins)
partition_range = (y_value_max - y_value_min)/No_of_bins


# In[11]:


vsmall_max = 3 + partition_range
small_max = vsmall_max + partition_range
mods_max = small_max + partition_range
modl_max = mods_max + partition_range
large_max = modl_max + partition_range
vlarge_max = large_max + partition_range


# In[12]:


meal_data = pd.read_csv("meal.csv", header = None)
vsmall_bin = []
small_bin = []
mods_bin = []
modl_bin = []
large_bin = []
vlarge_bin = []


# In[ ]:





# 

# In[13]:


bins = []
for i in range(len(y_meal_values)):
        if y_meal_values[i] < vsmall_max:
            vsmall_bin.append(meal_data.iloc[i])
            bins.append(0)
        elif y_meal_values[i] < small_max:
            small_bin.append(meal_data.iloc[i])
            bins.append(1)
        elif y_meal_values[i] < mods_max:
            mods_bin.append(meal_data.iloc[i])
            bins.append(2)
        elif y_meal_values[i] < modl_max:
            modl_bin.append(meal_data.iloc[i])
            bins.append(3)
        elif y_meal_values[i] < large_max:
            large_bin.append(meal_data.iloc[i])
            bins.append(4)
        else:
            vlarge_bin.append(meal_data.iloc[i])
            bins.append(5)


# In[14]:


sum = len(vsmall_bin) + len(small_bin) + len(mods_bin) + len(modl_bin)+len(large_bin)+len(vlarge_bin)


# In[15]:


Feature_Extraction.main()


# In[16]:


feature_matrix = pd.read_csv("mealDataFeatures.csv")
feature_matrix.to_numpy()


# In[17]:


kmeans = KMeans(n_clusters=6, init='k-means++').fit(feature_matrix)
k_labels = kmeans.labels_


# In[18]:


kmeans_sse = kmeans.inertia_
kmeans_entropy, kmeans_purity = entropy_calc(bins, k_labels)
print(kmeans_sse, kmeans_entropy, kmeans_purity)


# In[19]:


dbscan = DBSCAN(eps=200, min_samples = 7).fit(feature_matrix)
dblabel = dbscan.labels_
db_entropy, db_purity = entropy_calc(bins, dblabel)
# print(dblabel)
print(db_entropy, db_purity)


# In[ ]:





# In[20]:


db1 = []
db2 = []
db3 = []
db4 = []
db5 = []
db6 = []
for i in range(len(feature_matrix)):
    if dblabel[i] == -1:
        db1.append(feature_matrix.iloc[i])
    if dblabel[i] == 0:
        db2.append(feature_matrix.iloc[i])
    if dblabel[i] == 1:
        db3.append(feature_matrix.iloc[i])
    if dblabel[i] == 2:
        db4.append(feature_matrix.iloc[i])
    if dblabel[i] == 3:
        db5.append(feature_matrix.iloc[i])
    if dblabel[i] == 4:
        db6.append(feature_matrix.iloc[i])


# In[21]:


db1_np = np.array(db1)
db1_center = db1_np.mean(axis = 0)
db2_np = np.array(db2)
db2_center = db2_np.mean(axis = 0)
db3_np = np.array(db3)
db3_center = db3_np.mean(axis = 0)
db4_np = np.array(db4)
db4_center = db4_np.mean(axis = 0)
db5_np = np.array(db5)
db5_center = db5_np.mean(axis = 0)
db6_np = np.array(db6)
db6_center = db6_np.mean(axis = 0)


# In[22]:


print(db6_center[1])


# In[ ]:





# In[23]:


db1_dist = np.sum(np.linalg.norm(db1-db1_center, axis=1))
db2_dist = np.sum(np.linalg.norm(db2-db2_center, axis=1))
db3_dist = np.sum(np.linalg.norm(db3-db3_center, axis=1))
db4_dist = np.sum(np.linalg.norm(db4-db4_center, axis=1))
db5_dist = np.sum(np.linalg.norm(db5-db5_center, axis=1))
db6_dist = np.sum(np.linalg.norm(db6-db6_center, axis=1))

db_sse = db1_dist + db2_dist + db3_dist + db4_dist + db5_dist + db6_dist
print(db_sse)


# In[24]:


result = [[kmeans_sse, db_sse,kmeans_entropy, db_entropy, kmeans_purity, db_purity]] 
result = pd.DataFrame(result, columns = ['SSE for Kmeans', 'SSE for DBScan', 'Entropy for Kmeans', 'Entropy for DBScan', 'Purity for Kmeans', 'Purity for DBScan'])
result.to_csv('C:/Users/aishu/Desktop/College/DM/assign 3/Results.csv',index = False)

