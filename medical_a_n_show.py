import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from sklearn.preprocessing import scale


df= pd.read_csv('KaggleV2-May-2016.csv')
#print(df.head())
#print(df.describe)
print(df.columns)
print(df.info())

df.columns= ['PatientId', 'AppointmentID', 'Gender', 'ScheduledDay',
       'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hypertension',
       'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received', 'No-show']

df['Age'] = df['Age'].clip(lower=0, upper=None)
df['No-show'] = df['No-show'].astype('category')
print(df.columns)

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

print(df.info())

print(df['No-show'].value_counts())
df['dayofweek'] = df['AppointmentDay'].dt.dayofweek
df = df.join(pd.get_dummies(df['dayofweek']))
print(df.columns)
df.rename(columns = {0:'Sunday',1: 'Monday',2: 'Tuesday',3: 'Wednesday', 4: 'Thursday', 5:'Friday', 6:'Saturday'}, inplace = True)
df.drop(['PatientId', 'AppointmentID'], 1, inplace= True)
for i in df.columns[5:]:
    print(i, sorted(df[i].unique()), '\n')
#.drop(['AppointmentDay', 'ScheduledDay'], 1)
df =df[(df.Age >= 0)]
df_days_diff= df.ScheduledDay- df.AppointmentDay
df_days_diff= df_days_diff.apply(lambda x: x.total_seconds()/(3600 * 24))

def calcHour(x):
    x=str(x)
    h= int(x[11:13])
    m= int(x[14:16])
    s= int(x[17:])
    return round(h+m/60+s/3600)


df['ScheduledHourOfTheDay'] = df.ScheduledDay.apply(calcHour)
print('ScheduledHourOfTheDay', sorted(df['ScheduledHourOfTheDay'].unique()))

x= 'No-show'
df_OnlyShow = df[df[x] == 'Yes']
df_= pd.DataFrame()
df_['Age'] = range(100)

print('Gender Analysis\n')
men = df_.Age.apply(lambda x: len(df_OnlyShow[(df_OnlyShow.Age == x) & (df_OnlyShow.Gender == 'M')]))
women= df_.Age.apply(lambda x: len(df_OnlyShow[(df_OnlyShow.Age == x) & (df_OnlyShow.Gender == 'F')]))

plt.barh(range(100), men, color= 'b')
plt.barh(range(100), women, color = 'r')
plt.legend(['M','F'])
plt.ylabel('Age')
    plt.title('Women visit the doctor more often')

plt.show()
men_women= np.array(men, women)
sns.barplot(men_women, range(100))
#x=range(100),
men_hyp=df_[df_.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'M') & (df.Hypertension== 1)]))
female_hyp= df_[df_.columns[0]].apply(lambda x: len(df[(df.Age== x) & (df.Gender== 'F') & (df.Hypertension== 1)]))
plt.figure(figsize = (10,10))
plt.subplot(2,2,1)
plt.barh(range(100),men_hyp/men)
plt.barh(range(100),female_hyp/women, color = 'r')
plt.title('Hypertension')
plt.legend(['M','F'], loc = 2)


men_diabetes =df_[df_.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'M') & (df.Diabetes== 1)]))
female_diabetes= df_[df_.columns[0]].apply(lambda x: len(df[(df.Age== x) & (df.Gender== 'F') & (df.Diabetes== 1)]))
plt.figure(figsize = (10,10))
plt.subplot(2,2,1)
plt.barh(range(100),men_diabetes/men)
plt.barh(range(100),female_diabetes/women, color = 'r')
plt.title('Diabetes')
plt.legend(['M','F'], loc = 2)

men_drink=df_[df_.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'M') & (df.Alcoholism== 1)]))
female_drink= df_[df_.columns[0]].apply(lambda x: len(df[(df.Age== x) & (df.Gender== 'F') & (df.Alcoholism== 1)]))
plt.figure(figsize = (10,10))
plt.subplot(2,2,1)
plt.barh(range(100),men_drink/men)
plt.barh(range(100),female_drink/women, color = 'r')
plt.title('Male Alcoholics visit more often')
plt.legend(['M','F'], loc = 2)

men_hc=df_[df_.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'M') & (df.Handicap== 1)]))
female_hc= df_[df_.columns[0]].apply(lambda x: len(df[(df.Age== x) & (df.Gender== 'F') & (df.Handicap== 1)]))
plt.figure(figsize = (10,10))
plt.subplot(2,2,1)
plt.barh(range(100),men_hc/men)
plt.barh(range(100),female_hc/women, color = 'r')
plt.title('Handicap')
plt.legend(['M','F'], loc = 2)

men_sms=df_[df_.columns[0]].apply(lambda x: len(df[(df.Age == x) & (df.Gender == 'M') & (df.SMS_received== 1)]))
female_sms= df_[df_.columns[0]].apply(lambda x: len(df[(df.Age== x) & (df.Gender== 'F') & (df.SMS_received== 1)]))
plt.figure(figsize = (10,10))
plt.subplot(2,2,1)
plt.barh(range(100),men_sms/men)
plt.barh(range(100),female_sms/women, color = 'r')
plt.title('SMS_Recieved')
plt.legend(['M','F'], loc = 2)




