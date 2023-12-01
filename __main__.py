#Omer Atilim Koca
#01.12.2023
#Time-Series prediction using Transformer

import os
import xlsxwriter as xlsxwriter
import time
from train_model import train_model
########################################################################################################################
# initialization

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seedRange       =   1                  # seed number will be changed till 'seedRange'
epochRunning    =   2                # epoch number for running
parameterNumber   =   4                   # 1 --> CGM only, 2 --> CGM + Basal Insulin, 3 --> CGM + Basal Insulin + CHO
layerNumber     =   1                    # number of model layers
modelType       =   2                    # 0 --> RNN, 1 --> LSTM, 2 --> GRU, 3 --> BiRNN, 4 --> BiLSTM, 5 --> BiGRU,
                                         # 6 --> ConvRNN, 7 --> ConvLSTM, 8 --> ConvGRU
testFlag        =   1                    # if test flag is 1, test code will run. If it is 0, it will not run.
plotFlag        =   0                    # if plot flag is 1, plots will appear. If it is 1, plots will not appear.
patientNumber   =   1                  # total number of patients
start           =   time.time()          # record start time
seedList        =   list(range(0,seedRange))
horizonList=[6,12,18,24]


wsList      = ['patient540', 'patient544', 'patient552', 'patient559', 'patient563', 'patient567', 'patient570',
               'patient575', 'patient584', 'patient588', 'patient591', 'patient596']

########################################################################################################################

#  Running simulation

for parameter in range(parameterNumber):
    workbook = xlsxwriter.Workbook("Transformer+  "f"{parameter+1}"+"_parameter_"  +"_.xlsx")
    for patientFlag in range(patientNumber):
        epochList = []
        epoch_min = []
        worksheet = workbook.add_worksheet(wsList[patientFlag])
        worksheet.write('A1', 'Seed Number')
        worksheet.write('B1', 'Epoch Val')
        worksheet.write('C1', 'Val Min')
        for i in range(seedRange):
            print(f"{wsList[patientFlag]} is in progress with seed number {i} ")
            min_val, epoch_val = train_model(i, epochRunning, modelType, testFlag, patientFlag, layerNumber, parameter+1, plotFlag,horizonList[parameter])
            epochList.append(epoch_val)
            epoch_min.append(min_val)
            worksheet.write_column(1, 0, seedList)
            worksheet.write_column(1, 1, epochList)
            worksheet.write_column(1, 2, epoch_min)

    workbook.close()





end=time.time()
print("The time of execution of above program is :",
      (end-start) /60, "m")