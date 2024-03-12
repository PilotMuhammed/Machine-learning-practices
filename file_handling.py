labopen = open("ICS.csv","r+")
#labopen = open("ICS.csv","r")
#labopen = open("ICS.csv","W")

#print(labopen.readable())
#print(labopen.readline())
#print(labopen.readlines())
print(labopen.readlines()[6])

#for i in labopen.readlines():
 #print(i)
labopen.close()

### by using pandas
import pandas as pd
df = pd.read_csv('ICS.csv')
print(df.head(1))
print(df)
print(df.head())
