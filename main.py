import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#price, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning
def CalculateMissingEntries(data, column, count):
    missingCount = 0
    for row in data:
        if row[column] == "NULL":
            missingCount += 1
    percentage = missingCount * 100 / count
    return percentage

def UniqueValueCount(data, column):
    uniqueValues = []
    count = 0
    for row in data:
        if row[column] not in uniqueValues:
            uniqueValues.append(row[column])
            count += 1
    return count

def FindMinValue(data, column):
    minValue = 100000000
    for row in data:
        if(row[column] != "NULL"):
            if(float(row[column]) < float(minValue)):
                minValue = row[column]
    return minValue

def FindMaxValue(data, column):
    maxValue = -100000000
    for row in data:
        if(row[column] != "NULL"):
            if(float(row[column]) > float(maxValue)):
                maxValue = row[column]
    return maxValue

def FindQuartile(data, column, quar):
    sortedArray = []
    elementCount = 0
    for row in data:
        if(row[column] != "NULL"):
            sortedArray.append(row[column])
            elementCount += 1
    sortedArray.sort()
    quartileValue = sortedArray[int(elementCount * quar)]
    return quartileValue

def CalculateAverage(data, column):
    sumOfAll = 0
    countOfAll = 0
    for row in data:
        if(row[column] != "NULL"):
            sumOfAll += float(row[column])
            countOfAll += 1
    average = float(sumOfAll / countOfAll)
    return average
        

def CalculateStandardDeviation(data, column):
    variance = 0
    average = CalculateAverage(data, column)
    lineCount = len(data)
    for row in data:
        if row[column] != "NULL":
            variance += (float(row[column]) - float(average)).__pow__(2)
    variance = variance / (lineCount - 1)
    stdev = math.sqrt(variance)
    return stdev

def FindFirstMode(data, column):
    valueArray = []
    occurencesCount = []
    for row in data:
        if row[column] != "NULL":
            if row[column] not in valueArray:
                valueArray.append(row[column])
                occurencesCount.append(1)
            else:
                occurencesCount[valueArray.index(row[column])] += 1
    maxValue = 0
    for value in occurencesCount:
        if value > maxValue:
            maxValue = value
    return valueArray[occurencesCount.index(maxValue)]

def FindSecondMode(data, column):
    valueArray = []
    occurencesCount = []
    for row in data:
        if row[column] != "NULL":
            if row[column] not in valueArray:
                valueArray.append(row[column])
                occurencesCount.append(1)
            else:
                occurencesCount[valueArray.index(row[column])] += 1
    lastMax = float('-inf')
    maxValue = 0
    for value in occurencesCount:
        if value > maxValue:
            lastMax = maxValue
            maxValue = value
        elif value > lastMax and value < maxValue:
            lastMax = value
    return valueArray[occurencesCount.index(lastMax)]


def FindRepeatingTable(data, index):
    valueArray = []
    occurencesCount = []
    for row in data:
        if row[index] not in valueArray:
            valueArray.append(row[index])
            occurencesCount.append(1)
        else:
            occurencesCount[valueArray.index(row[index])] += 1

    maximum_value = 0
    for value in occurencesCount:
        if value > maximum_value:
            maximum_value = value

    valueArray.append(occurencesCount)

    return valueArray

def FindTablePercentages(data, column):
    valueArray = []
    occurencesCount = []
    for row in data:
        if row[column] not in valueArray:
            valueArray.append(row[column])
            occurencesCount.append(1)
        else:
            occurencesCount[valueArray.index(row[column])] += 1

    maximum_value = 0
    for value in occurencesCount:
        if value > maximum_value:
            maximum_value = value

    valueArray.append(occurencesCount)

    occurencesCount = valueArray.__len__() - 1

    for i in range(valueArray.__len__() - 1):
        valueArray[occurencesCount][i] = valueArray[occurencesCount][i] * 100 / data.__len__()

    return valueArray

def TotalCount(data, column):
    count = 0
    for row in data:
        if(row[column] != "NULL"):
            count += 1
    return count
def PrintTable(attribute, lineCount, missingPercentage, kardinalumas, maxValue, minValue,
                     medianQ2, Q1, Q3, average, variance):
    
    print(attribute)
    print(lineCount)
    print(missingPercentage)
    print(kardinalumas)
    print(maxValue)
    print(minValue)
    print(medianQ2)
    print(Q1)
    print(Q3)
    print(average)
    print(variance)
    print('----------------------------------')

def printCategoryAtribute(attribute, count, missingEntries, kardinalumas, firstMode, firstModeRepeating,
                              firstModePercentage, secondMode, secondModeRepeating, secondModePercentage):
    print(attribute)
    print(count)
    print(missingEntries)
    print(kardinalumas)
    print(firstMode)
    print(firstModeRepeating)
    print(firstModePercentage)
    print(secondMode)
    print(secondModeRepeating)
    print(secondModePercentage)
    print('----------------------------------')

def FixMissingEntryNumerical(data, column, average):
    for row in data:
        if(row[column] == "NULL"):
            row[column] = average
    return

def FixMissingEntryCategoric(data, column, mode):
    for row in data:
        if(row[column] == "NULL"):
            row[column] = mode
    return

def CorrectExtremeValues(data, column, firstQ, thirdQ):
    upperBound = float(thirdQ) + 1.5 * (float(thirdQ) - float(firstQ))
    lowerBound = float(firstQ) - 1.5 * (float(thirdQ) - float(firstQ))
    for i in range(len(data)):
        if(float(data[i][column]) > upperBound):
            data[i][column] = upperBound
        if(float(data[i][column]) < lowerBound):
            data[i][column] = lowerBound
    return data

with open('Housing.csv') as file:
    read = csv.reader(file, delimiter=',')
    selectedData = [] #House data
    lineCount = 0
    for row in read:
        selectedDataRow = []
        for i in range(0,10):
            if row[i] != "":
                selectedDataRow.append(row[i])
            else:
                selectedDataRow.append("NULL")
        selectedData.append(selectedDataRow)
        lineCount += 1

# Tolydiniai atributai:
#House price:
atribute = "House price"
count = TotalCount(selectedData, 0)
missingEntries = CalculateMissingEntries(selectedData, 0, count)
kardinalumas = UniqueValueCount(selectedData, 0)
maxValue = FindMaxValue(selectedData, 0)
minValue = FindMinValue(selectedData, 0)
quartile1 = FindQuartile(selectedData, 0, 0.25)
median = FindQuartile(selectedData, 0, 0.5)
quartile3 = FindQuartile(selectedData, 0, 0.75)
average = CalculateAverage(selectedData, 0)
variance = CalculateStandardDeviation(selectedData, 0)
PrintTable(atribute, count, missingEntries, kardinalumas, maxValue, minValue, median,
                 quartile1, quartile3, average, variance)
FixMissingEntryNumerical(selectedData,0,average)
CorrectExtremeValues(selectedData, 0, quartile1, quartile3)

#Area:
atribute = "Area"
count = TotalCount(selectedData, 1)
missingEntries = CalculateMissingEntries(selectedData, 1, count)
kardinalumas = UniqueValueCount(selectedData, 1)
maxValue = FindMaxValue(selectedData, 1)
minValue = FindMinValue(selectedData, 1)
quartile1 = FindQuartile(selectedData, 1, 0.25)
median = FindQuartile(selectedData, 1, 0.5)
quartile3 = FindQuartile(selectedData, 1, 0.75)
average = CalculateAverage(selectedData, 1)
variance = CalculateStandardDeviation(selectedData, 1)
PrintTable(atribute, count, missingEntries, kardinalumas, maxValue, minValue, median,
                 quartile1, quartile3, average, variance)
FixMissingEntryNumerical(selectedData,1,average)
CorrectExtremeValues(selectedData, 1, quartile1, quartile3)

#bedrooms:
atribute = "Bedrooms"
count = TotalCount(selectedData, 2)
missingEntries = CalculateMissingEntries(selectedData, 2, count)
kardinalumas = UniqueValueCount(selectedData, 2)
maxValue = FindMaxValue(selectedData, 2)
minValue = FindMinValue(selectedData, 2)
quartile1 = FindQuartile(selectedData, 2, 0.25)
median = FindQuartile(selectedData, 2, 0.5)
quartile3 = FindQuartile(selectedData, 2, 0.75)
average = CalculateAverage(selectedData, 2)
variance = CalculateStandardDeviation(selectedData, 2)
PrintTable(atribute, count, missingEntries, kardinalumas, maxValue, minValue, median,
                 quartile1, quartile3, average, variance)
FixMissingEntryNumerical(selectedData,2,average)
CorrectExtremeValues(selectedData, 2, quartile1, quartile3)

#-----------------------

#Kategoriniai atributai:
#bathrooms:
atribute = "Bathrooms"
count = TotalCount(selectedData, 3)
missingEntries = CalculateMissingEntries(selectedData, 3, count)
kardinalumas = UniqueValueCount(selectedData, 3)
firstMode = FindFirstMode(selectedData, 3)
modeRepeatingTable = FindRepeatingTable(selectedData, 3)
modeTablePercentages = FindTablePercentages(selectedData, 3)
modeRepeatingTable[kardinalumas].sort()
modeTablePercentages[kardinalumas].sort()
modeRepeating = modeRepeatingTable[kardinalumas][kardinalumas - 1]
modePercentage = modeTablePercentages[kardinalumas][kardinalumas - 1]
secondMode = FindSecondMode(selectedData, 3)
modeRepeatingSecond = modeRepeatingTable[kardinalumas][kardinalumas - 2]
modePercentageSecond = modeTablePercentages[kardinalumas][kardinalumas - 2]
printCategoryAtribute(atribute, count, missingEntries, kardinalumas, firstMode, modeRepeating, modePercentage, secondMode,
                      modeRepeatingSecond, modePercentageSecond)
FixMissingEntryCategoric(selectedData, 3, firstMode)

#Stories:
atribute = "Stories"
count = TotalCount(selectedData, 4)
missingEntries = CalculateMissingEntries(selectedData,4, count)
kardinalumas = UniqueValueCount(selectedData, 4)
firstMode = FindFirstMode(selectedData, 4)
modeRepeatingTable = FindRepeatingTable(selectedData, 4)
modeTablePercentages = FindTablePercentages(selectedData, 4)
modeRepeatingTable[kardinalumas].sort()
modeTablePercentages[kardinalumas].sort()
modeRepeating = modeRepeatingTable[kardinalumas][kardinalumas - 1]
modePercentage = modeTablePercentages[kardinalumas][kardinalumas - 1]
secondMode = FindSecondMode(selectedData, 4)
modeRepeatingSecond = modeRepeatingTable[kardinalumas][kardinalumas - 2]
modePercentageSecond = modeTablePercentages[kardinalumas][kardinalumas - 2]
printCategoryAtribute(atribute, count, missingEntries, kardinalumas, firstMode, modeRepeating, modePercentage, secondMode,
                      modeRepeatingSecond, modePercentageSecond)
FixMissingEntryCategoric(selectedData, 4, firstMode)
#Mainroad
atribute = "Mainroad"
count = TotalCount(selectedData, 5)
missingEntries = CalculateMissingEntries(selectedData, 5, count)
kardinalumas = UniqueValueCount(selectedData, 5)
firstMode = FindFirstMode(selectedData, 5)
modeRepeatingTable = FindRepeatingTable(selectedData, 5)
modeTablePercentages = FindTablePercentages(selectedData, 5)
modeRepeatingTable[kardinalumas].sort()
modeTablePercentages[kardinalumas].sort()
modeRepeating = modeRepeatingTable[kardinalumas][kardinalumas - 1]
modePercentage = modeTablePercentages[kardinalumas][kardinalumas - 1]
secondMode = FindSecondMode(selectedData, 5)
modeRepeatingSecond = modeRepeatingTable[kardinalumas][kardinalumas - 2]
modePercentageSecond = modeTablePercentages[kardinalumas][kardinalumas - 2]
printCategoryAtribute(atribute, count, missingEntries, kardinalumas, firstMode, modeRepeating, modePercentage, secondMode,
                      modeRepeatingSecond, modePercentageSecond)
FixMissingEntryCategoric(selectedData, 5, firstMode)
#Guestroom
count = TotalCount(selectedData, 6)
missingEntries = CalculateMissingEntries(selectedData, 6, count)
kardinalumas = UniqueValueCount(selectedData, 6)
firstMode = FindFirstMode(selectedData, 6)
modeRepeatingTable = FindRepeatingTable(selectedData, 6)
modeTablePercentages = FindTablePercentages(selectedData, 6)
modeRepeatingTable[kardinalumas].sort()
modeTablePercentages[kardinalumas].sort()
modeRepeating = modeRepeatingTable[kardinalumas][kardinalumas - 1]
modePercentage = modeTablePercentages[kardinalumas][kardinalumas - 1]
secondMode = FindSecondMode(selectedData, 6)
modeRepeatingSecond = modeRepeatingTable[kardinalumas][kardinalumas - 2]
modePercentageSecond = modeTablePercentages[kardinalumas][kardinalumas - 2]
printCategoryAtribute('Guestroom', count, missingEntries, kardinalumas, firstMode, modeRepeating, modePercentage, secondMode,
                      modeRepeatingSecond, modePercentageSecond)
FixMissingEntryCategoric(selectedData, 6, firstMode)
#Basement
count = TotalCount(selectedData, 7)
missingEntries = CalculateMissingEntries(selectedData, 7, count)
kardinalumas = UniqueValueCount(selectedData, 7)
firstMode = FindFirstMode(selectedData, 7)
modeRepeatingTable = FindRepeatingTable(selectedData, 7)
modeTablePercentages = FindTablePercentages(selectedData, 7)
modeRepeatingTable[kardinalumas].sort()
modeTablePercentages[kardinalumas].sort()
modeRepeating = modeRepeatingTable[kardinalumas][kardinalumas - 1]
modePercentage = modeTablePercentages[kardinalumas][kardinalumas - 1]
secondMode = FindSecondMode(selectedData, 7)
modeRepeatingSecond = modeRepeatingTable[kardinalumas][kardinalumas - 2]
modePercentageSecond = modeTablePercentages[kardinalumas][kardinalumas - 2]
printCategoryAtribute('Basement', count, missingEntries, kardinalumas, firstMode, modeRepeating, modePercentage, secondMode,
                      modeRepeatingSecond, modePercentageSecond)
FixMissingEntryCategoric(selectedData, 7, firstMode)
#Hot water heating
count = TotalCount(selectedData, 8)
missingEntries = CalculateMissingEntries(selectedData, 8, count)
kardinalumas = UniqueValueCount(selectedData, 8)
firstMode = FindFirstMode(selectedData, 8)
modeRepeatingTable = FindRepeatingTable(selectedData, 8)
modeTablePercentages = FindTablePercentages(selectedData, 8)
modeRepeatingTable[kardinalumas].sort()
modeTablePercentages[kardinalumas].sort()
modeRepeating = modeRepeatingTable[kardinalumas][kardinalumas - 1]
modePercentage = modeTablePercentages[kardinalumas][kardinalumas - 1]
secondMode = FindSecondMode(selectedData, 8)
modeRepeatingSecond = modeRepeatingTable[kardinalumas][kardinalumas - 2]
modePercentageSecond = modeTablePercentages[kardinalumas][kardinalumas - 2]
printCategoryAtribute('Hot water heating', count, missingEntries, kardinalumas, firstMode, modeRepeating, modePercentage, secondMode,
                      modeRepeatingSecond, modePercentageSecond)
FixMissingEntryCategoric(selectedData, 8, firstMode)

#Air conditioning
count = TotalCount(selectedData, 9)
missingEntries = CalculateMissingEntries(selectedData, 9, count)
kardinalumas = UniqueValueCount(selectedData, 9)
firstMode = FindFirstMode(selectedData, 9)
modeRepeatingTable = FindRepeatingTable(selectedData, 9)
modeTablePercentages = FindTablePercentages(selectedData, 9)
modeRepeatingTable[kardinalumas].sort()
modeTablePercentages[kardinalumas].sort()
modeRepeating = modeRepeatingTable[kardinalumas][kardinalumas - 1]
modePercentage = modeTablePercentages[kardinalumas][kardinalumas - 1]
secondMode = FindSecondMode(selectedData, 9)
modeRepeatingSecond = modeRepeatingTable[kardinalumas][kardinalumas - 2]
modePercentageSecond = modeTablePercentages[kardinalumas][kardinalumas - 2]
printCategoryAtribute('Air conditioning', count, missingEntries, kardinalumas, firstMode, modeRepeating, modePercentage, secondMode,
                      modeRepeatingSecond, modePercentageSecond)
FixMissingEntryCategoric(selectedData, 9, firstMode)
#-----------------------
dataArray = []
for row in selectedData:
    #price - 1, area - 2, bedrooms - 3, bathrooms - 4, 
    # stories - 5, etc.
    dataArray.append(row[9])


x = np.array(dataArray)
plt.hist(x, density=False, bins=int(1 + 3.22 * np.log(x.size)))
plt.show()

prices = []
areas = []
bedrooms = []  
bathrooms = []
stories = []
mainroad = []
guestroom = []
basement = []
hotwater = []
airconditioning = []
for i in range(selectedData.__len__()):
    prices.append(float(selectedData[i][0]))
    areas.append(float(selectedData[i][1]))
    bedrooms.append(float(selectedData[i][2]))
    bathrooms.append(float(selectedData[i][3]))
    stories.append(float(selectedData[i][4]))
    mainroad.append(str(selectedData[i][5]))
    guestroom.append(str(selectedData[i][6]))
    basement.append(str(selectedData[i][7]))
    hotwater.append(str(selectedData[i][8]))
    airconditioning.append(str(selectedData[i][9]))



plt.scatter(areas, prices)
plt.title("Area to price")
data = {'areas': areas, 
        'prices': prices,
        'bedrooms': bedrooms,
        }
df = pd.DataFrame(data)
correlation = df['areas'].corr(df['prices'])
print(correlation)
plt.show()

plt.scatter(bedrooms, areas)
plt.title("Bedrooms to area")
df = pd.DataFrame(data)
correlation = df['bedrooms'].corr(df['areas'])
print(correlation)
plt.show()

plt.scatter(bedrooms, prices)
plt.title("Bedrooms to price")
df = pd.DataFrame(data)
correlation = df['bedrooms'].corr(df['prices'])
print(correlation)
plt.show()

data = {'areas': areas, 
        'prices': prices,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms, 
        'stories': stories,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwater': hotwater,
        'airconditioning': airconditioning
        }
pd.plotting.scatter_matrix(df)
plt.show()

#bar plots
sns.countplot(x="bathrooms", hue="stories", data=data)
plt.title('Bathrooms per story')
plt.show()

sns.countplot(x="stories", hue="mainroad", data=data)
plt.title('Multiple story buildings near main roads')
plt.show()

#histograms
sns.histplot(data=data, x="prices", hue="stories", multiple="stack", kde=True)
plt.title('Price for stories')
plt.show()

sns.histplot(data=data, x="prices", hue="airconditioning", multiple="stack", kde=True)
plt.title('Price for airconditioning')
plt.show()
 
# box plot
sns.boxplot(data=data, x="stories", y="prices")
plt.title('Stories for price')
plt.show()

sns.boxplot(data=data, x="airconditioning", y="prices")
plt.title('Price for airconditioning')
plt.show()
df = pd.DataFrame(data)
cov_matrix = df[['prices', 'areas', 'bedrooms']].cov()
print(f'{cov_matrix}\n')
 
# koreliacijos matrica
corr_matrix = df[['prices', 'areas', 'bedrooms']].corr()
print(corr_matrix)
 
# grafikas
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Koreliacijos matrica')
plt.show()

scaler = MinMaxScaler()
normalized = pd.DataFrame(scaler.fit_transform(df[['prices', 'areas', 'bedrooms']]), columns=df[['prices', 'areas', 'bedrooms']].columns)
 
print(normalized['bedrooms'])