# -*- coding:utf-8 -*-

from BayesianNetworks import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#############################
## Example Tests from Bishop `Pattern Recognition and Machine Learning` textbook on page 377
#############################
BatteryState = readFactorTable(['battery'], [0.9, 0.1], [[1, 0]])
FuelState = readFactorTable(['fuel'], [0.9, 0.1], [[1, 0]])
GaugeBF = readFactorTable(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9], [[1, 0], [1, 0], [1, 0]])
# print(BatteryState)
# print(FuelState)
carNet = [BatteryState, FuelState, GaugeBF] # carNet is a list of factors 
## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
joinFactors(joinFactors(BatteryState, FuelState), GaugeBF)
joinFactors(joinFactors(GaugeBF, FuelState), BatteryState)

marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'gauge')
joinFactors(marginalizeFactor(GaugeBF, 'gauge'), BatteryState)

joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState)
marginalizeFactor(joinFactors(joinFactors(GaugeBF, FuelState), BatteryState), 'battery')

marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'gauge')
marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'fuel')

evidenceUpdateNet(carNet, ['fuel', 'battery'], [1, 0])

# inference
print("inference starts")
print(inference(carNet, ['battery', 'fuel'], [], []) )        ## chapter 8 equation (8.30)
print(inference(carNet, ['battery'], ['fuel'], [0]))           ## chapter 8 equation (8.31)
print(inference(carNet, ['battery'], ['gauge'], [0]))          ##chapter 8 equation  (8.32)
print(inference(carNet, [], ['gauge', 'battery'], [0, 0]))    ## chapter 8 equation (8.33)
print("inference ends")
###########################################################################
#RiskFactor Data Tests
###########################################################################
riskFactorNet = pd.read_csv('RiskFactorsData.csv')

# Create factors

income      = readFactorTablefromData(riskFactorNet, ['income'])
smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit    = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up     = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'income'])
diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])

## you need to create more factor tables

risk_net = [income, smoke, long_sit, stay_up, exercise, bmi, diabetes]
print("income dataframe is ")
print(income)
factors = riskFactorNet.columns

# example test p(diabetes|smoke=1,exercise=2,long_sit=1)

margVars = list(set(factors) - {'diabetes', 'smoke', 'exercise','long_sit'})
obsVars  = ['smoke', 'exercise','long_sit']
obsVals  = [1, 2, 1]

p = inference(risk_net, margVars, obsVars, obsVals)
print(p)

 
###########################################################################
# Please write your own test script
# HW3 test scripts start from here
###########################################################################

print("=============================================\n\n")
print("=============================================\n\n")

print("Problem 1\n")
# create factors
income      = readFactorTablefromData(riskFactorNet, ['income'])
smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])
exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
long_sit    = readFactorTablefromData(riskFactorNet, ['long_sit', 'income'])
stay_up     = readFactorTablefromData(riskFactorNet, ['stay_up', 'income'])
bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'exercise', 'income', 'long_sit'])
bp          = readFactorTablefromData(riskFactorNet, ['bp', 'exercise', 'long_sit', 'income', 'stay_up', 'smoke'])
cholest     = readFactorTablefromData(riskFactorNet, ['cholesterol', 'exercise', 'stay_up', 'income', 'smoke'])
stroke       = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol'])
attack      = readFactorTablefromData(riskFactorNet, ['attack', 'bmi', 'bp', 'cholesterol'])
angina      = readFactorTablefromData(riskFactorNet, ['angina', 'bmi', 'bp', 'cholesterol'])
diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])

# build the risk net
risk_net = [income, smoke, exercise, long_sit, stay_up, bmi, diabetes, bp, cholest, stroke, attack, angina]
# all the factors
factors = set(riskFactorNet.columns)

# Problem 2
print("Problem 2\n")
for i, disease in enumerate(('diabetes', 'stroke', 'attack', 'angina')):
    print(f'{disease}')
    print('habits')
    margVars = list(factors - set([disease, 'smoke', 'exercise','long_sit', 'stay_up']))
    obsVars  = ['smoke', 'exercise', 'long_sit', 'stay_up']
    obsVals  = [1, 2, 1, 1]
    print(inference(risk_net, margVars, obsVars, obsVals))

    obsVals  = [2, 1, 2, 2]
    print(inference(risk_net, margVars, obsVars, obsVals))

    print('health')
    margVars = list(factors - set([disease, 'bp', 'cholesterol','bmi']))
    obsVars  = ['bp', 'cholesterol','bmi']
    obsVals = [1, 1, 3]
    print(inference(risk_net, margVars, obsVars, obsVals))

    obsVals = [3, 2, 2]
    print(inference(risk_net, margVars, obsVars, obsVals))

print("=============================================\n\n")

# Problem 3
print("Problem 3")
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,12))
for i, disease in enumerate(['diabetes', 'stroke', 'attack', 'angina']):
    prob = []
    margVars = list(factors - {disease, 'income'})
    for income in range(1, 9):
        result = inference(risk_net, margVars, ["income"], [income])
        prob.append(result["probs"][0])

    plt.subplot(2, 2, i+1)
    plt.plot(np.arange(1, 9), prob, color='palevioletred', linestyle='-.', marker='s', linewidth=2.0, ms=8.0, alpha=1.0)
    plt.xlabel("Income Status", size = 20)
    plt.ylabel(f"Probability of having {disease}", size = 20)
    plt.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig("problem3.png")
print("=============================================\n\n")

# Problem 4
print("Problem 4\n")
income2      = readFactorTablefromData(riskFactorNet, ['income'])
stroke2      = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol', 'exercise', 'smoke'])
attack2      = readFactorTablefromData(riskFactorNet, ['attack', 'bmi', 'bp', 'cholesterol', 'exercise', 'smoke'])
angina2      = readFactorTablefromData(riskFactorNet, ['angina', 'bmi', 'bp', 'cholesterol', 'exercise', 'smoke'])
diabetes2    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi', 'exercise', 'smoke'])

# build the new risk net
risk_net = [income2, smoke, exercise, long_sit, stay_up, bmi, bp, cholest, stroke2, attack2, angina2, diabetes2]
# all the factors
factors = set(riskFactorNet.columns)

for disease in ('diabetes', 'stroke', 'attack', 'angina'):
    print(f"{disease} ")
    margVars = list(factors - set([disease, 'smoke', 'exercise','long_sit', 'stay_up']))
    obsVars  = ['smoke', 'exercise', 'long_sit', 'stay_up']
    obsVals  = [1, 2, 1, 1]
    print(inference(risk_net, margVars, obsVars, obsVals))

    obsVals  = [2, 1, 2, 2]
    print(inference(risk_net, margVars, obsVars, obsVals))

    margVars = list(factors - set([disease, 'bp', 'cholesterol','bmi']))
    obsVars  = ['bp', 'cholesterol','bmi']
    obsVals = [1, 1, 3]
    print(inference(risk_net, margVars, obsVars, obsVals))

    obsVals = [3, 2, 2]
    print(inference(risk_net, margVars, obsVars, obsVals))
    print("\n")

print("=============================================\n\n")

############################### task 5
print("Problem 5 (without edges)")
margVars = list(factors - {"diabetes", "stroke"})
print(inference(risk_net, margVars, ['diabetes'], [1]))
print(inference(risk_net, margVars, ['diabetes'], [3]))
print('\n')

print("Problem 5 (with edges)")
new_income          = readFactorTablefromData(riskFactorNet, ['income'])
new_stroke_2        = readFactorTablefromData(riskFactorNet, ['stroke', 'bmi', 'bp', 'cholesterol', 'exercise', 'smoke', 'diabetes'])
risk_net = [new_income, smoke, exercise, long_sit, stay_up, bmi, diabetes2, bp, cholest, new_stroke_2, attack2, angina2]

margVars = list(factors - {"diabetes", "stroke"})
print(inference(risk_net, margVars, ['diabetes'], [1]))
print(inference(risk_net, margVars, ['diabetes'], [3]))

print("All the problems are finished\n\n")
 