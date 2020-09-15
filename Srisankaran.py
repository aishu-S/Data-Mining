#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from datetime import datetime


# In[7]:


cgm = pd.read_csv("CGMData.csv")
insulin = pd.read_csv("InsulinData.csv")
insulin['Alarm'] = insulin['Alarm'].astype('string')
insulin['Time'] = insulin['Time'].astype('string')
insulin['Date'] = insulin['Date'].astype('string')
cgm['Time'] = cgm['Time'].astype('string')
cgm['Date'] = cgm['Date'].astype('string')


# In[19]:


date_stamp = ""
time_stamp1 = ""


# 

# In[28]:


i= len(insulin)-1
while 0<i:
    alrm = insulin.at[i, 'Alarm']
    if str(alrm) == "AUTO MODE ACTIVE PLGM OFF" :
        date_stamp=insulin.at[i, 'Date']
        time_stamp1 = insulin.at[i, 'Time']
        break;
    i-=1

temp = str(time_stamp1).replace(':', '')
time_stamp = int(temp)
print(time_stamp)
print(temp)
print(date_stamp)


# In[65]:


# Manual
m_hg_dt = 0
m_hg_critical_dt = 0
m_inrange_dt = 0
m_range_sec_dt = 0
m_hg_lev1_dt = 0
m_hg_lev2_dt = 0

m_hg_on = 0
m_hg_critical_on = 0
m_inrange_on = 0
m_range_sec_on = 0
m_hg_lev1_on = 0
m_hg_lev2_on = 0

m_hg_wd = 0
m_hg_critical_wd = 0
m_inrange_wd = 0
m_range_sec_wd = 0
m_hg_lev1_wd = 0
m_hg_lev2_wd = 0

break_index = 0
count = 1
fin = 0
i = len(cgm)-1
while 0<i:
    fin+=1
    a1 = cgm.at[i, 'Time']
    a1 = a1.replace(':','')
    a = int(str(a1))
    b = cgm.at[i, 'Date']
    b1 = cgm.at[i-1, 'Date']
    c1 = cgm.at[i-1, 'Time']
    c1 = c1.replace(':','')
    c = int(str(c1))
    if (str(b) == date_stamp and a<time_stamp and c>time_stamp):
        break_index+= i
        break;
    if b!=b1:
        count+=1
    time = cgm.at[i, 'Time']
    # Daytime
    if '06:00:00' < str(time) <= '23:59:59':
        cgmlevel = cgm.at[i, 'Sensor Glucose (mg/dL)']
        if cgmlevel > 180:
            m_hg_dt+=1
            m_hg_wd+=1
        if cgmlevel >250:
            m_hg_critical_dt+=1
            m_hg_critical_wd+=1
        if cgmlevel >= 70 and cgmlevel <= 180:
            m_inrange_dt+=1
            m_inrange_wd+=1
        if cgmlevel >= 70 and cgmlevel <= 150:
            m_range_sec_dt+=1
            m_range_sec_wd+=1
        if cgmlevel < 70:
            m_hg_lev1_dt+=1
            m_hg_lev1_wd+=1
        if cgmlevel < 54:
            m_hg_lev2_dt+=1
            m_hg_lev2_wd+=1
    #Over night
    else:
        cgmlevel = cgm.at[i, 'Sensor Glucose (mg/dL)']
        if cgmlevel > 180:
            m_hg_on+=1
            m_hg_wd+=1
        if cgmlevel >250:
            m_hg_critical_on+=1
            m_hg_critical_wd+=1
        if cgmlevel >= 70 and cgmlevel <= 180:
            m_inrange_on+=1
            m_inrange_wd+=1
        if cgmlevel >= 70 and cgmlevel <= 150:
            m_range_sec_on+=1
            m_range_sec_wd+=1
        if cgmlevel < 70:
            m_hg_lev1_on+=1
            m_hg_lev1_wd+=1
        if cgmlevel < 54:
            m_hg_lev2_on+=1
            m_hg_lev2_wd+=1
    i-=1
print("Daytime")
print(m_hg_dt, m_hg_critical_dt, m_inrange_dt, m_range_sec_dt, m_hg_lev1_dt, m_hg_lev2_dt)
print(m_hg_on, m_hg_critical_on, m_inrange_on, m_range_sec_on, m_hg_lev1_on, m_hg_lev2_on)
print(m_hg_wd, m_hg_critical_wd, m_inrange_wd, m_range_sec_wd, m_hg_lev1_wd, m_hg_lev2_wd)
print(break_index)
print(count)
print(fin)


# In[51]:


tot = count * 288 / 100


# In[68]:


per_m_hg_dt = m_hg_dt / tot
per_m_hg_critical_dt = m_hg_critical_dt / tot
per_m_inrange_dt = m_inrange_dt / tot
per_m_range_sec_dt = m_range_sec_dt / tot
per_m_hg_lev1_dt = m_hg_lev1_dt /tot
per_m_hg_lev2_dt = m_hg_lev2_dt /tot

per_m_hg_on = m_hg_on / tot
per_m_hg_critical_on = m_hg_critical_on / tot
per_m_inrange_on = m_inrange_on / tot
per_m_range_sec_on = m_range_sec_on / tot
per_m_hg_lev1_on = m_hg_lev1_on /tot
per_m_hg_lev2_on = m_hg_lev2_on /tot

per_m_hg_wd = m_hg_wd / tot
per_m_hg_critical_wd = m_hg_critical_wd / tot
per_m_inrange_wd = m_inrange_wd / tot
per_m_range_sec_wd = m_range_sec_wd / tot
per_m_hg_lev1_wd = m_hg_lev1_wd /tot
per_m_hg_lev2_wd = m_hg_lev2_wd /tot

print(per_m_hg_dt, per_m_hg_critical_dt, per_m_inrange_dt, per_m_range_sec_dt, per_m_hg_lev1_dt, per_m_hg_lev2_dt)
print(per_m_hg_on, per_m_hg_critical_on, per_m_inrange_on, per_m_range_sec_on, per_m_hg_lev1_on, per_m_hg_lev2_on)
print(per_m_hg_wd, per_m_hg_critical_wd, per_m_inrange_wd, per_m_range_sec_wd, per_m_hg_lev1_wd, per_m_hg_lev2_wd)


# In[70]:


# Auto mode
a_hg_dt = 0
a_hg_critical_dt = 0
a_inrange_dt = 0
a_range_sec_dt = 0
a_hg_lev1_dt = 0
a_hg_lev2_dt = 0

a_hg_on = 0
a_hg_critical_on = 0
a_inrange_on = 0
a_range_sec_on = 0
a_hg_lev1_on = 0
a_hg_lev2_on = 0

a_hg_wd = 0
a_hg_critical_wd = 0
a_inrange_wd = 0
a_range_sec_wd = 0
a_hg_lev1_wd = 0
a_hg_lev2_wd = 0

count1 = 1
fin2 = 0

for i in range(break_index, 0, -1):
    fin2+=1
    time = cgm.at[i, 'Time']
    if i!=0:
        d1 = cgm.at[i, 'Date']
        d2 = cgm.at[i-1, 'Date']
        if d1!=d2:
            count1+=1
    # Daytime
    if '06:00:00' < str(time) <= '23:59:59':
        cgmlevel = cgm.at[i, 'Sensor Glucose (mg/dL)']
        if cgmlevel > 180:
            a_hg_dt+=1
            a_hg_wd+=1
        if cgmlevel >250:
            a_hg_critical_dt+=1
            a_hg_critical_wd+=1
        if cgmlevel >= 70 and cgmlevel <= 180:
            a_inrange_dt+=1
            a_inrange_wd+=1
        if cgmlevel >= 70 and cgmlevel <= 150:
            a_range_sec_dt+=1
            a_range_sec_wd+=1
        if cgmlevel < 70:
            a_hg_lev1_dt+=1
            a_hg_lev1_wd+=1
        if cgmlevel < 54:
            a_hg_lev2_dt+=1
            a_hg_lev2_wd+=1
    #Over night
    else:
        cgmlevel = cgm.at[i, 'Sensor Glucose (mg/dL)']
        if cgmlevel > 180:
            a_hg_on+=1
            a_hg_wd+=1
        if cgmlevel >250:
            a_hg_critical_on+=1
            a_hg_critical_wd+=1
        if cgmlevel >= 70 and cgmlevel <= 180:
            a_inrange_on+=1
            a_inrange_wd+=1
        if cgmlevel >= 70 and cgmlevel <= 150:
            a_range_sec_on+=1
            a_range_sec_wd+=1
        if cgmlevel < 70:
            a_hg_lev1_on+=1
            a_hg_lev1_wd+=1
        if cgmlevel < 54:
            a_hg_lev2_on+=1
            a_hg_lev2_wd+=1
    i-=1
print(a_hg_dt, a_hg_critical_dt, a_inrange_dt, a_range_sec_dt, a_hg_lev1_dt, a_hg_lev2_dt)
print(a_hg_on, a_hg_critical_on, a_inrange_on, a_range_sec_on, a_hg_lev1_on, a_hg_lev2_on)
print(a_hg_wd, a_hg_critical_wd, a_inrange_wd, a_range_sec_wd, a_hg_lev1_wd, a_hg_lev2_wd)
tot1 = count1 * 288 /100
print(tot1)
print(count1)
print(fin2)
fin3 = fin+fin2
print(fin3)


# In[71]:


per_a_hg_dt = a_hg_dt / tot1
per_a_hg_critical_dt = a_hg_critical_dt / tot1
per_a_inrange_dt = a_inrange_dt / tot1
per_a_range_sec_dt = a_range_sec_dt / tot1
per_a_hg_lev1_dt = a_hg_lev1_dt /tot1
per_a_hg_lev2_dt = a_hg_lev2_dt /tot1

per_a_hg_on = a_hg_on / tot1
per_a_hg_critical_on = a_hg_critical_on / tot1
per_a_inrange_on = a_inrange_on / tot1
per_a_range_sec_on = a_range_sec_on / tot1
per_a_hg_lev1_on = a_hg_lev1_on /tot1
per_a_hg_lev2_on = a_hg_lev2_on /tot1

per_a_hg_wd = a_hg_wd / tot1
per_a_hg_critical_wd = a_hg_critical_wd / tot1
per_a_inrange_wd = a_inrange_wd / tot1
per_a_range_sec_wd = a_range_sec_wd / tot1
per_a_hg_lev1_wd = a_hg_lev1_wd /tot1
per_a_hg_lev2_wd = a_hg_lev2_wd /tot1

print(per_a_hg_dt, per_a_hg_critical_dt, per_a_inrange_dt, per_a_range_sec_dt, per_a_hg_lev1_dt, per_a_hg_lev2_dt)
print(per_a_hg_on, per_a_hg_critical_on, per_a_inrange_on, per_a_range_sec_on, per_a_hg_lev1_on, per_a_hg_lev2_on)
print(per_a_hg_wd, per_a_hg_critical_wd, per_a_inrange_wd, per_a_range_sec_wd, per_a_hg_lev1_wd, per_a_hg_lev2_wd)

r = per_a_hg_wd+ per_a_hg_critical_wd+ per_a_inrange_wd+ per_a_range_sec_wd+ per_a_hg_lev1_wd+ per_a_hg_lev2_wd
print(r)


# In[74]:


result = [[per_m_hg_on, per_m_hg_critical_on, per_m_inrange_on, per_m_range_sec_on, per_m_hg_lev1_on, per_m_hg_lev2_on, per_m_hg_dt, per_m_hg_critical_dt, per_m_inrange_dt, per_m_range_sec_dt, per_m_hg_lev1_dt, per_m_hg_lev2_dt, per_m_hg_wd, per_m_hg_critical_wd, per_m_inrange_wd, per_m_range_sec_wd, per_m_hg_lev1_wd, per_m_hg_lev2_wd], [ per_a_hg_on, per_a_hg_critical_on, per_a_inrange_on, per_a_range_sec_on, per_a_hg_lev1_on, per_a_hg_lev2_on, per_a_hg_dt, per_a_hg_critical_dt, per_a_inrange_dt, per_a_range_sec_dt, per_a_hg_lev1_dt, per_a_hg_lev2_dt, per_a_hg_wd, per_a_hg_critical_wd, per_a_inrange_wd, per_a_range_sec_wd, per_a_hg_lev1_wd, per_a_hg_lev2_wd]] 

result_df = pd.DataFrame(result, columns=['Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)', 'Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)','Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
                                         'Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)', 'Over night percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)',
                                         'Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)','Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)', 'Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)','Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
                                         'Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)', 'Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)',
                                         'Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)','Whole day Percentage time in hyperglycemia (CGM > 180 mg/dL)', 'Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)','Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
                                         'Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)', 'Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)',
                                         'Whole day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'], index=['Manual mode', 'Auto mode'])
print(result_df)
result_df.to_csv('Results.csv')


# In[43]:


print(cgm.shape[0])
print(insulin.shape[0])

