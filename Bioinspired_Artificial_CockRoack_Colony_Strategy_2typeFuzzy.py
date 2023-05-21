from scipy.spatial import distance
#from gekko import GEKKO
import numpy as np
#from scipy import ndimage
#import maximum_filter_findPeak
import math
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import copy
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
from numpy import random
from scipy import signal
from scipy import ndimage
from scipy import misc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import copy
import sys
#import subprocess
import signal
#import potentialFieldPlanner15
import itertools
from datetime import date
from pyit2fls import IT2FS_Gaussian_UncertStd,Mamdani,IT2Mamdani, IT2FLS,R_IT2FS_Gaussian_UncertStd, \
    L_IT2FS_Gaussian_UncertStd, \
                     min_t_norm, product_t_norm, max_s_norm, IT2FS_plot,crisp
from numpy import  array, linspace


robspeed=1
STEPS_PER_EPISODE = 960#150 #100
CLEAN_RAY = 3
CLUSTER_RAY = 1
N_totalX=172
N_totalY=100
N_robots=4
show_animation = False
TESTING = True
N_Walkman=500 #1500
update_period = 60
Lowlim_N_Walk=int(N_Walkman-N_Walkman*30/100)

############################################################################
StartingSubArea0_X=[140, 112,127,139,  166,139,   65,112 ]
StartingSubArea0_Y=[ 20,  0 , 0, 0,   0,  14,     25, 16]


EndingSubArea0_Y=[  24,  5, 16, 3,  13,  22,     44, 22]
EndingSubArea0_X=[  112,126,138,166,  168,168,   74, 139]   




                     # 0k8  ok9   ok10 ok10bis  ok19 ok20 ok21 ok22 ok23
StartingSubArea1_X=[  150,  53,     64,  62,     62,    72 , 106, 44, 166]
StartingSubArea1_Y=[ 50,  45,      45, 59 ,     62,    74 ,62,  62, 25   ]

EndingSubArea1_Y = [  59, 59,      59, 62 ,     78,    78 ,73,  73,  44]   

EndingSubArea1_X=[    52, 63,     115, 115,    72,    116 ,115, 52, 170]

                    



                     # ok11 ok12   ok13   ok14   
StartingSubArea2_X=[  100, 125,  127, 169]
StartingSubArea2_Y=[  46,  45,  63,  63 ]


EndingSubArea2_X=[   125,  168,  133, 170]   
                     

EndingSubArea2_Y=[   57,   60, 74,  74 ]


                    # ok15 ok16  ok17   ok  ok
StartingSubArea3_X=[  168,    11,   53,   64, 127]
StartingSubArea3_Y=[  53,  74,   78,   74, 74]

EndingSubArea3_X=[   10,    52,   168,  115, 169 ]   

                    

                    

EndingSubArea3_Y=[    99,  99 ,  99,   77, 77]


StartListX=[StartingSubArea0_X,StartingSubArea1_X,StartingSubArea2_X,StartingSubArea3_X]
StartListY=[StartingSubArea0_Y,StartingSubArea1_Y,StartingSubArea2_Y,StartingSubArea3_Y]




###################################################################
d=0
settembre2=   [211,241,295,321,357,407,446,501,551,587,624,692,724,758,797]
settembre7=   [211,241,295,321,357,407,446,501,551,587,624,692,724,758,797]
settembre8=   [211,241,295,321,357,407,446,501,551,587,624,692,724,758,797]
settembre9=   [211,241,295,321,357,407,446,501,551,587,624,692,724,758,797]
settembre10=  [211,241,295,321,357,407,446,501,551,587,624,692,724,758,797]
settembre11=  [211,241,295,321,357,407,446,501,551,587,624,692,724,758,797]
settembre12=  [211,241,295,321,357,407,446,501,551,587,624,692,724,758,797]


Giorni_New=['6_settembre/','7_settembre/','8_settembre/','9_settembre/','10_settembre/','11_settembre/','12_settembre/']
#Matrici_New=[settembre6,settembre7,settembre8,settembre9,settembre10,settembre11,settembre12]

###################################################################





SubFerormone=np.ones((N_totalX,N_totalY))/10 # da verificare la sintassi di np.ones
# nodes:            0    1   2   3   4   5   6    7   8   9   10    11   12   13   14  15  16  17  18  19

                    #ok1 #ok2  #ok3  #ok4  #ok5  #ok6  #ok7


 

# List_MyDestinationNodeChoice,Center_SubArea= myFood_Input_Fuzzy1(NewUpgrade,DQNAgents,env,Max_Steps):


def myFood_Input_Fuzzy2(List_MyChoice,current_dark_choice, current_feromone_choice,DQNAgents,env,inx,iny,robs,Center_SubArea):
     

    #for k in range(N_robots):

    
        # ho bisogno di 2 conti con Fuzzy.
        # il primo è la scelta della sub-area.
        # una volta scelta l'area di interesse, devo azzerare il valore della relativa area, per evitare che
        #tutti i robot scelgano la stessa.

    cont=0
    cont2=0
        
        #MyPath=[]
        #MyPath2=[]
        #myboolPath=True
    #List_step_n_h_k = []
    List_step = []

    MinDistRob=MinDistanceRob(robs, DQNAgents,env,inx,iny)

            
    distanceFromDestinationNode = np.hypot(inx - Center_SubArea[List_MyChoice[robs]][0], iny -Center_SubArea[robs][1])
    

    
    #for n in range(len(List_MyChoice)): #lista dei nodi prioritari, ogni elemento è una lista di nodi, in numero di elementi è pari al numero di robot; 
            #for k in range(len(current_dark_choice)): # 
                #for h in range(len(current_feromone_choice)):
    #type2Fuzzy_Control2                    (MyChoiceNode,       distanceFromDestinationNode,current_dark_choice, current_feromone_choice,MinDistRob):
    Next_step_Evaluation=type2Fuzzy_Control2(env,robs,List_MyChoice[robs],distanceFromDestinationNode,current_dark_choice, current_feromone_choice,MinDistRob)
                    
        #List_step_n_h_k.append([Next_step_Evaluation,k])
    #    List_step.append([Next_step_Evaluation])
        

    #max_item = max(List_step)
            
    #print("max_item",max_item)
    #MaxCurrent_Choice=List_step.index(max_item) #
    #MychoiceCurrent=List_step_n_h_k[MaxCurrent_Choice][1]
            
                      

           
        
        #List_Next_Node.sort(reverse=True)
        #List_MyChoice.append[Mychoice]
        #print("Mychoice",Mychoice)
        #cMeno[(Starting[k], Mychoice) ]=10000
        #cPiu[(Starting[k], Mychoice) ]=0


    
    return Next_step_Evaluation

def MinDistanceRob(MyRob,DQNAgents,env,inx,iny):
    
    List_Distance_Between_Robs=[]
    
    for robs in range(N_robots ):
        
        if robs!=MyRob:

            
            Onedistance = np.hypot(DQNAgents[robs].positionRobX[-1] - inx, DQNAgents[robs].positionRobY[-1] - iny)
        
            #print("distances[i][j]",Onedistance)
        
            List_Distance_Between_Robs.append(Onedistance)

    MinDistRob= min(List_Distance_Between_Robs)
    print("List_Distance_Between_Robs,MinDistRob",List_Distance_Between_Robs,MinDistRob)
    return MinDistRob






def type2Fuzzy_Control2(env,robs,MyChoiceNode,distanceFromDestinationNode,current_dark_choice, current_feromone_choice,MinDistRob):
    
    x1=current_feromone_choice
    #cMeno=12.4
    #cPiu=200
    #x1=((200-cMeno)/200) +0.1
    #x1=cMeno/200
    #x2=cPiu/200
    Old_inx=env.DQNAgents[robs].positionRobX.copy()
    Old_iny=env.DQNAgents[robs].positionRobY.copy()
    if(len(Old_inx)>2):
        if (Old_inx[-1]==Old_inx[-2]) and (Old_iny[-1]==Old_iny[-2]) or (Old_inx[-1]==Old_inx[-3]) and (Old_iny[-1]==Old_iny[-3]):
            x1=0


    x2=current_dark_choice
    x3= distanceFromDestinationNode
    x4=MinDistRob 
    print("x1",x1,"x2",x2,"x3",x3,"x4",x4)
    #input("enter")
    #Defining the domain of the output
    #domain = linspace(1., 0.15, 10)
    domain = linspace(0., 1., 100000) #ok

    # Defining the domain of the input variable Goal_manhattan_Grid,
    domain1 = linspace(0., 9., 100000) #ok ok

    # Defining the domain of the input variable d_Euclidean,
    domain2 = linspace(0., 9., 100000) #ok

    # Defining the domain of the input variable MyGrid priority.
    domain3 = linspace(0., 200, 10000) #ok

    # Defining the domain of the input variable MyGrid priority.
    domain4 = linspace(0., 200, 10000) #ok
    #The params input for IT2FS_Gaussian_UncertStd function, is a list consisting of the mean, the standard deviation center, the standard deviation spread, and the height of the set

    # Defining the Small set for the input variable Goal_manhattan_Grid.
    #R_IT2FS_Gaussian_UncertStd
    Small1 = IT2FS_Gaussian_UncertStd(domain1, [0., 0.8, 0.4, 1.])

    # Defining the Small set for the input variable d_Euclidean.
    #Small2 = IT2FS_Gaussian_UncertStd(domain, [0.4, 0.15, 0.05, 1.])

    # Defining the Small set for the input variable MyGrid priority.
    Small2 = IT2FS_Gaussian_UncertStd(domain2, [0., 0.8, 0.4, 1.])

    Small3 = IT2FS_Gaussian_UncertStd(domain3, [0., 15., 10., 1.])

    Small4 = IT2FS_Gaussian_UncertStd(domain4, [0., 15., 10., 1.])

    # Defining the Medium set for the input variable Goal_manhattan_Grid.
    Medium1 = IT2FS_Gaussian_UncertStd(domain1, [4., 0.8, 0.4, 1.])

    # Defining the Medium set for the input variable d_Euclidean.
    #Medium2 = IT2FS_Gaussian_UncertStd(domain, [0.5, 0.15, 0.05, 1.])

    # Defining the Medium set for the input variable MyGrid priority.
    Medium2 = IT2FS_Gaussian_UncertStd(domain2, [4., 0.8, 0.4, 1.])

    Medium3 = IT2FS_Gaussian_UncertStd(domain3, [100., 15., 10., 1.])

    Medium4 = IT2FS_Gaussian_UncertStd(domain4, [100., 15., 10., 1.])


    # Defining the Large set for the input variable Goal_manhattan_Grid.
    Large1 = IT2FS_Gaussian_UncertStd(domain1, [9., 0.8, 0.4, 1.])

    # Defining the Large set for the input variable d_Euclidean.
    #Large2 = IT2FS_Gaussian_UncertStd(domain, [0.6, 0.15, 0.05, 1.])

    # Defining the Large set for the input variable MyGrid priority.
    Large2 = IT2FS_Gaussian_UncertStd(domain2, [9., 0.8, 0.4, 1.])

    Large3 = IT2FS_Gaussian_UncertStd(domain3, [200., 15., 10., 1.])

    Large4 = IT2FS_Gaussian_UncertStd(domain4, [200., 15., 10., 1.])

    # Plotting the sets defined for the input variable x1.
    #IT2FS_plot(Small1, Medium1, Large1, legends=["Small", "Medium", "large"])

    # Plotting the sets defined for the input variable x2.
    #IT2FS_plot(Small2, Medium2, Large2, legends=["Small", "Medium", "large"])


    # Plotting the sets defined for the input variable x2.
    #IT2FS_plot(Small3, Medium3, Large3, legends=["Small", "Medium", "large"])

    # Defining the mamdani interval type 2 fuzzy logic system
    myIT2FLS = IT2Mamdani(min_t_norm, max_s_norm, method="Centroid", algorithm="KM")
    #myIT2FLS = Mamdani(min_t_norm, max_s_norm)

    # Adding the input variables to the myIT2FLS
    myIT2FLS.add_input_variable("x1")
    myIT2FLS.add_input_variable("x2")
    myIT2FLS.add_input_variable("x3")
    myIT2FLS.add_input_variable("x4")
    # Adding the output variables to the myIT2FLS


    LB = IT2FS_Gaussian_UncertStd(domain, [0., 0.02, 0.01, 1.])
    LM = IT2FS_Gaussian_UncertStd(domain, [0.11, 0.02, 0.01, 1.])
    LA = IT2FS_Gaussian_UncertStd(domain, [0.22, 0.02, 0.01, 1.])

    MB = IT2FS_Gaussian_UncertStd(domain, [0.33, 0.02, 0.01, 1.])
    MM = IT2FS_Gaussian_UncertStd(domain, [0.44, 0.02, 0.01, 1.])
    MA = IT2FS_Gaussian_UncertStd(domain, [0.55, 0.02, 0.01, 1.])

    HB = IT2FS_Gaussian_UncertStd(domain, [0.66, 0.02, 0.01, 1.])
    HM = IT2FS_Gaussian_UncertStd(domain, [0.77, 0.02, 0.01, 1.])
    HA = IT2FS_Gaussian_UncertStd(domain, [0.88, 0.02, 0.01, 1.])
    HAA = IT2FS_Gaussian_UncertStd(domain, [1., 0.02, 0.01, 1.])

    #IT2FS_plot(NB, NM, ZZ, PM, PB, legends=["Negative Big", "Negative Medium", 
    #                                       "Zero", "Positive Medium", 
    #                                       "Positive Big"], filename="delay_pid_output_sets")

    '''
    # The Small set is defined as a Guassian IT2FS with uncertain standard deviation 
    # value. The mean, the standard deviation center, the standard deviation spread, 
    # and the height of the set are set to 0., 0.15, 0.1, and 1., respectively.
    Small = L_IT2FS_Gaussian_UncertStd(domain, [0, 0.15, 0.05, 1.])

    # The Medium set is defined as a Guassian IT2FS with uncertain standard deviation 
    # value. The mean, the standard deviation center, the standard deviation spread, 
    # and the height of the set are set to 0.5, 0.15, 0.1, and 1., respectively.
    Medium = L_IT2FS_Gaussian_UncertStd(domain, [0.5, 0.15, 0.05, 1.])

    # The Large set is defined as a Guassian IT2FS with uncertain standard deviation 
    # value. The mean, the standard deviation center, the standard deviation spread, 
    # and the height of the set are set to 1., 0.15, 0.1, and 1., respectively.
    Large = L_IT2FS_Gaussian_UncertStd(domain, [1., 0.15, 0.05, 1.])
    '''


    myIT2FLS.add_output_variable("y1")  
    #x1=current_feromone_choice
    #x2=current_dark_choice #Priorities
    #x3= distanceFromDestinationNode
    #x4=MinDistRob 

                      #feromone_choice       #distanceFromD #Priorities     #MinDistRob 
    myIT2FLS.add_rule([("x1", Small1), ("x3", Small3),("x2", Small2), ("x4", Large4)], [("y1",LA )]) #

    myIT2FLS.add_rule([("x1", Medium1), ("x3", Small3),("x2", Small2), ("x4", Large4)], [("y1", LM)])

    myIT2FLS.add_rule([("x1", Large1), ("x3", Medium3),("x2", Small2), ("x4", Large4)], [("y1", LB)])

    
    myIT2FLS.add_rule([("x1", Small1), ("x3", Medium3),("x2", Medium2), ("x4", Large4)], [("y1", MA)]) #

    myIT2FLS.add_rule([("x1", Medium1), ("x3",Large3),("x2", Medium2), ("x4", Medium4)], [("y1", MM)])

    myIT2FLS.add_rule([("x1", Large1), ("x3", Large3),("x2", Medium2), ("x4", Medium4)], [("y1",MB )])

    
    myIT2FLS.add_rule([("x1", Small1), ("x3", Small3),("x2", Large2), ("x4", Medium4)], [("y1",HA )]) #

    myIT2FLS.add_rule([("x1", Medium1), ("x3", Small3),("x2", Large2), ("x4", Medium4)], [("y1", HM)])

    myIT2FLS.add_rule([("x1", Large1), ("x3", Medium3),("x2", Large2), ("x4", Small4)], [("y1", HB)])


    myIT2FLS.add_rule([("x1", Small1), ("x3", Medium3),("x2", Small2), ("x4", Small4)], [("y1", LA)]) #

    myIT2FLS.add_rule([("x1", Medium1), ("x3", Large3),("x2", Small2), ("x4", Small4)], [("y1", LM)])

    myIT2FLS.add_rule([("x1", Large1), ("x3", Large3),("x2", Small2), ("x4", Small4)], [("y1",LB )])


    myIT2FLS.add_rule([("x1", Small1), ("x3", Small3),("x2", Medium2), ("x4", Large4)], [("y1",MA )])

    myIT2FLS.add_rule([("x1", Medium1), ("x3", Small3),("x2", Medium2), ("x4", Large4)], [("y1", MM)])

    myIT2FLS.add_rule([("x1", Large1), ("x3", Medium3),("x2", Medium2), ("x4", Large4)], [("y1", MB)])

    
    myIT2FLS.add_rule([("x1", Small1), ("x3", Medium3),("x2", Large2), ("x4", Large4)], [("y1", HA)])

    myIT2FLS.add_rule([("x1", Medium1), ("x3", Large3),("x2", Large2), ("x4", Medium4)], [("y1", HM)])

    myIT2FLS.add_rule([("x1", Large1), ("x3", Large3),("x2", Large2), ("x4", Medium4)], [("y1",HB )])


    myIT2FLS.add_rule([("x1", Small1), ("x3", Small3),("x2", Small2), ("x4", Medium4)], [("y1",LA )])

    myIT2FLS.add_rule([("x1", Medium1), ("x3", Small3),("x2", Small2), ("x4", Medium4)], [("y1", LM)])

    myIT2FLS.add_rule([("x1", Large1), ("x3", Medium3),("x2", Small2), ("x4", Small4)], [("y1", LB)])


    myIT2FLS.add_rule([("x1", Small1), ("x3", Medium3),("x2", Medium2), ("x4", Small4)], [("y1", MA)]) 

    myIT2FLS.add_rule([("x1", Medium1), ("x3",Large3),("x2", Medium2), ("x4", Small4)], [("y1", MM)])

    myIT2FLS.add_rule([("x1", Large1), ("x3", Large3),("x2", Medium2), ("x4", Small4)], [("y1",MB )])


    myIT2FLS.add_rule([("x1", Small1), ("x3", Small3),("x2", Large2), ("x4", Large4)], [("y1", HAA)])

    myIT2FLS.add_rule([("x1", Medium1), ("x3", Small3),("x2", Large2), ("x4", Large4)], [("y1", HM)])

    myIT2FLS.add_rule([("x1", Large1), ("x3", Medium3),("x2", Large2), ("x4", Large4)], [("y1", HB)])



    myIT2FLS.add_rule([("x1", Small1), ("x3", Medium3),("x2", Small2), ("x4", Large4)], [("y1",LA )])

    myIT2FLS.add_rule([("x1", Medium1), ("x3", Large3),("x2", Small2), ("x4", Medium4)], [("y1", LM)])

    myIT2FLS.add_rule([("x1", Large1), ("x3", Large3),("x2", Small2), ("x4", Medium4)], [("y1",LB )])


    myIT2FLS.add_rule([("x1", Small1), ("x3", Small3),("x2", Medium2), ("x4", Medium4)], [("y1", MA)])

    myIT2FLS.add_rule([("x1", Medium1), ("x3", Small3),("x2", Medium2), ("x4", Medium4)], [("y1", MM)])

    myIT2FLS.add_rule([("x1", Large1), ("x3", Medium3),("x2", Medium2), ("x4", Small4)], [("y1", MB)])


    myIT2FLS.add_rule([("x1", Small1), ("x3", Medium3),("x2", Large2), ("x4", Small4)], [("y1",HA )])
    
    myIT2FLS.add_rule([("x1", Medium1), ("x3", Large3),("x2", Large2), ("x4", Small4)], [("y1", HM)])

    myIT2FLS.add_rule([("x1", Large1), ("x3", Large3),("x2", Large2), ("x4", Small4)], [("y1",HB )])


    #myIT2FLS.add_rule([("x1", Small1), ("x3", Small3),("x2", Small2), ("x4",Small4)], [("y1",LA )]) 
    
    #myIT2FLS.add_rule([("x1", Medium1), ("x3", Medium3),("x2", Medium2), ("x4", Medium4)], [("y1", MM)])

    #myIT2FLS.add_rule([("x1", Large1), ("x3", Large3),("x2", Large2), ("x4", Large4)], [("y1",HM )])



    #2   2   2   2   2222    LM
    myIT2FLS.add_rule([("x1", Medium1), ("x3",Medium3 ),("x2",Medium2 ), ("x4",Medium4 )], [("y1", LM)])
    #3   2   2   2   3222    LB
    myIT2FLS.add_rule([("x1", Large1), ("x3",Medium3 ),("x2",Medium2 ), ("x4", Medium4)], [("y1", LB)])
    #1   3   2   2   1322    MA
    myIT2FLS.add_rule([("x1",Small1 ), ("x3",Large3 ),("x2", Medium2), ("x4",Medium4 )], [("y1", MA)])
    #2   3   3   2   2332    MM
    myIT2FLS.add_rule([("x1",Medium1 ), ("x3",Large3 ),("x2",Large2 ), ("x4",Medium4 )], [("y1", MM)])
    #3   1   3   1   3131    MB
    myIT2FLS.add_rule([("x1",Large1 ), ("x3",Small3 ),("x2",Large2 ), ("x4",Small4 )], [("y1",MB )])
    #1   1   3   1   1131    HA
    myIT2FLS.add_rule([("x1",Small1 ), ("x3", Small3),("x2", Large2), ("x4",Small4 )], [("y1", HA)])
    #2   2   1   1   2211    HM
    myIT2FLS.add_rule([("x1",Medium1 ), ("x3", Medium3),("x2", Small2), ("x4",Small4 )], [("y1",HM )])
    #3   2   1   1   3211    HB
    myIT2FLS.add_rule([("x1",Large1 ), ("x3",Medium3 ),("x2",Small2 ), ("x4", Small4)], [("y1",HB )])
    #1   3   1   3   1313    LA
    myIT2FLS.add_rule([("x1",Small1 ), ("x3",Large3 ),("x2",Small2 ), ("x4",Large4 )], [("y1",LA )])
    #2   3   2   3   2323    LM
    myIT2FLS.add_rule([("x1", Medium1), ("x3",Large3 ),("x2",Medium2 ), ("x4",Large4 )], [("y1", LM)])
    #3   1   2   3   3123    LB
    myIT2FLS.add_rule([("x1",Large1 ), ("x3",Small3 ),("x2",Medium2 ), ("x4",Large4 )], [("y1",LB )])
    #1   1   2   3   1123    MA
    myIT2FLS.add_rule([("x1", Small1), ("x3",Small3 ),("x2", Medium2), ("x4",Large4 )], [("y1",MA )])
    #2   2   3   2   2232    MM
    myIT2FLS.add_rule([("x1",Medium1 ), ("x3",Medium3 ),("x2",Large2 ), ("x4", Medium4)], [("y1", MM)])
    #3   2   3   2   3232    MB
    myIT2FLS.add_rule([("x1", Large1), ("x3",Medium3 ),("x2",Large2 ), ("x4",Medium4 )], [("y1", MB)])
    #1   3   3   2   1332    HA
    myIT2FLS.add_rule([("x1", Small1), ("x3",Large3 ),("x2", Large2), ("x4", Medium4)], [("y1",HA )])
    #2   3   1   2   2312    HM
    myIT2FLS.add_rule([("x1", Medium1), ("x3",Large3 ),("x2", Small2), ("x4",Medium4 )], [("y1",HM )])
    #3   1   1   1   3111    HB
    myIT2FLS.add_rule([("x1",Large1 ), ("x3", Small3),("x2",Small2 ), ("x4",Small4 )], [("y1", HB)])
    #1   1   1   1   1111    LA
    myIT2FLS.add_rule([("x1",Small1 ), ("x3",Small3 ),("x2",Small2 ), ("x4",Small4 )], [("y1", LA)])
    #2   2   2   1   2221    LM
    myIT2FLS.add_rule([("x1",Medium1 ), ("x3", Medium3),("x2",Medium2 ), ("x4",Small4 )], [("y1", LM)])
    #3   2   2   1   3221    LB
    myIT2FLS.add_rule([("x1",Large1 ), ("x3",Medium3 ),("x2",Medium2 ), ("x4", Small4)], [("y1", LB)])
    #1   3   2   3   1323    MA
    myIT2FLS.add_rule([("x1", Small1), ("x3",Large3 ),("x2",Medium2 ), ("x4",Large4 )], [("y1",MA )])
    #2   3   3   3   2333    MM
    myIT2FLS.add_rule([("x1",Medium1 ), ("x3", Large3),("x2",Large2 ), ("x4",Large4 )], [("y1",MM )])
    #3   1   3   3   3133    MB
    myIT2FLS.add_rule([("x1", Large1), ("x3",Small3 ),("x2",Large2 ), ("x4",Large4 )], [("y1", MB)])
    #1   1   3   3   1133    HA
    myIT2FLS.add_rule([("x1", Small1), ("x3",Small3 ),("x2",Large2 ), ("x4", Large4)], [("y1", HA)])
    #2   2   1   2   2212    HM
    myIT2FLS.add_rule([("x1", Medium1), ("x3", Medium3),("x2", Small2), ("x4", Medium4)], [("y1",HM )])
    #3   2   1   2   3212    HB
    myIT2FLS.add_rule([("x1", Large1), ("x3",Medium3 ),("x2",Small2 ), ("x4", Medium4)], [("y1",HB )])
    #1   3   1   2   1312    LA
    myIT2FLS.add_rule([("x1",Small1 ), ("x3", Large3),("x2",Small2 ), ("x4",Medium4 )], [("y1", LA)])
    #2   3   2   2   2322    LM
    myIT2FLS.add_rule([("x1",Medium1 ), ("x3",Large3 ),("x2",Medium2 ), ("x4", Medium4)], [("y1",LM )])
    #3   1   2   1   3121    LB
    myIT2FLS.add_rule([("x1", Large1), ("x3", Small3),("x2",Medium2 ), ("x4",Small4 )], [("y1",LB )])
    #1   1   2   1   1121    MA
    myIT2FLS.add_rule([("x1", Small1), ("x3",Small3 ),("x2",Medium2 ), ("x4",Small4 )], [("y1",MA )])
    #2   2   3   1   2231    MM
    myIT2FLS.add_rule([("x1",Medium1 ), ("x3",Medium3 ),("x2",Large2 ), ("x4",Small4 )], [("y1",MM )])
    #3   2   3   1   3231    MB
    myIT2FLS.add_rule([("x1",Large1 ), ("x3",Medium3 ),("x2",Large2 ), ("x4", Small4)], [("y1",MB )])
    #1   3   3   3   1333    HA
    myIT2FLS.add_rule([("x1",Small1 ), ("x3",Large3 ),("x2", Large2), ("x4",Large4 )], [("y1",HA )])
    #2   3   1   3   2313    HM
    myIT2FLS.add_rule([("x1", Medium1), ("x3",Large3 ),("x2",Small2 ), ("x4",Large4 )], [("y1",HM )])
    #3   1   1   3   3113    HB
    myIT2FLS.add_rule([("x1",Large1 ), ("x3", Small3),("x2",Small2 ), ("x4",Large4 )], [("y1", HB)])
    #1   1   1   3   1113    LA
    myIT2FLS.add_rule([("x1",Small1 ), ("x3",Small3 ),("x2",Small2 ), ("x4", Large4)], [("y1", LA)])
    #3   3   3   1   3331    MB
    myIT2FLS.add_rule([("x1",Large1 ), ("x3", Large3),("x2", Large2), ("x4", Small4)], [("y1", MB)])
    #1   3   3   1   1331    HA
    myIT2FLS.add_rule([("x1",Small1 ), ("x3",Large3 ),("x2",Large2 ), ("x4",Small4 )], [("y1",HA )])
    #2   1   3   1   2131    HM
    myIT2FLS.add_rule([("x1",Medium1 ), ("x3",Small3 ),("x2",Large2 ), ("x4",Small4 )], [("y1", HM)])
    #3   1   1   1   3111    HB
    myIT2FLS.add_rule([("x1",Large1 ), ("x3", Small3),("x2", Small2), ("x4", Small4)], [("y1", HB)])
    #1   2   1   3   1213    LA
    myIT2FLS.add_rule([("x1",Small1 ), ("x3", Medium3),("x2",Small2 ), ("x4",Large4 )], [("y1", LA)])
    #2   2   1   3   2213    LM
    myIT2FLS.add_rule([("x1",Medium1 ), ("x3",Medium3 ),("x2", Small2), ("x4",Large4 )], [("y1",LM )])
    #3   3   2   3   3323    LB
    myIT2FLS.add_rule([("x1", Large1), ("x3",Large3 ),("x2",Medium2 ), ("x4",Large4 )], [("y1",LB )])
    #1   3   2   3   1323    MA
    myIT2FLS.add_rule([("x1",Small1 ), ("x3", Large3),("x2", Medium2), ("x4", Large4)], [("y1",MA )])
    #2   1   2   2   2122    MM
    myIT2FLS.add_rule([("x1", Medium1), ("x3",Small3 ),("x2", Medium2), ("x4",Medium4 )], [("y1",MM )])















    c, TR = myIT2FLS.evaluate({"x1":x1, "x2":x2,"x3":x3, "x4":x4})
    #c, TR = myIT2FLS.evaluate({"x1":x3, "x2":0.745})
    print("TR",TR)
    print("shape TR",np.shape(TR))
    print("c",c)
    o = TR["y1"]
    print("o",o)
    o1 = (TR["y1"][0] + TR["y1"][1] ) / 2
    print("o1",o1)
        

    #input("enter")
        

    
    return o1





def changeVisualization(signalNumber, frame):
    global TESTING

    if TESTING:
        TESTING = False
    else:
        TESTING = True

    print('(SIGQUIT) MAP-PLOTTING CHANGED TO ', TESTING)
    return

def doNothing(signalNumber, frame):
    return



class DQNAgent:
    def __init__(self):
        self.positionRobX = []
        self.positionRobY = []

        self.current_n_node= 0
        self.current_C= []
        self.EvenOdd=True
        self.done=False


    def my_Agent_action_update(self,NewpositionX,NewpositionY): #invariato
        print("sono nel my_Agent_action_update")
                
        # Update Q value for given state

        # And append to our training data
        self.positionRobX.append(NewpositionX)
        self.positionRobY.append(NewpositionY)

    def my_Agent_action_updateX(self,NewpositionX): #invariato
        print("sono nel my_Agent_action_updateX")
                
        # Update Q value for given state

        # And append to our training data
        self.positionRobX.append(NewpositionX)
        
    def my_Agent_action_updateY(self,NewpositionY): #invariato
        print("sono nel my_Agent_action_updateY")
                
        # Update Q value for given state

        # And append to our training data
            
        self.positionRobY.append(NewpositionY)

class BlobEnv:
    def __init__(self,DQNAgents):
        self.Data=np.zeros((N_totalX,N_totalY,2))
        self.RealWorld=np.zeros((N_totalX,N_totalY))
        self.Wall_Matrix=np.zeros((N_totalX,N_totalY))
        self.DQNAgents=DQNAgents
        self.Number_bugs=[]
        self.Starting=[]
        self.NewUpgradeBin=[]
        self.Wall_Bin=[]
        self.Center_SubArea=[]
        self.List_Distance_Between_RobsAndNodes=[]
        self.List_Distance_Between_RobsAndNodes_forEveryRobs=[]

    def reset(self,robs,StartX,StartY):
        print("sono nel reset")
                
        #### here I need to put to zero, heatmap, robot positions
        self.robs=robs
        self.Walls_and_Pillars_Matrix()  
        #### initialize the data matrix with dummy values only for testing
        self.Data=np.zeros((N_totalX,N_totalY,2))

        self.episode_step = 0
        #self.state_first=copy.deepcopy(self.Data)
        #self.first_sum = np.sum(self.state_first[:, :, 0]) 
        self.updateHeatmap(0)

        #self.DQNAgents[self.robs].my_Agent_action_update(StartX,StartY)
        self.DQNAgents[self.robs].positionRobX.append(StartX)
        self.DQNAgents[self.robs].positionRobY.append(StartY)
        self.PositionRobotMatrix()
        
        self.rewards_log = np.zeros((100,2))
        self.steps = 0
        self.reward = 0
        self.cumulative_reward = 0
    




    def updateHeatmap(self,count):
        print("updateHeatmap")
        #if timestep%5==0:
        #for robs in range(self.N_Walkman):
              
        #    self.PositionWalkmanMatrix(robs)
        #count
        #Dir='Test_all_Day1/MyHeatmap'+str(count)+'.csv'
        myFrame1='frame'+str(settembre2[count])+'.csv'
     
        #print("count",count)
        #NewUpgrade=cv2.imread("Testheatmap2/MyHeatmap"+str(count)+".csv") 
        #NewUpgrade=pd.read_csv(Dir,header=None)
        #print("nonetype",type(NewUpgrade),np.shape(NewUpgrade)) 
        MyState=np.zeros((N_totalX,N_totalY))
        d=0
        MyState=pd.read_csv(Giorni_New[d]+myFrame1,header=0,index_col=0)

        super_threshold_indices11 = MyState == 0.9
        MyState[super_threshold_indices11] = 0.75

        super_threshold_indices22 = MyState == 0.8
        MyState[super_threshold_indices22] = 0.50

        super_threshold_indices33 = MyState == 0.7
        MyState[super_threshold_indices33] = 0.25              
        
        self.Data[:, :, 0]=self.Data[:,:,0]+np.array(MyState)

        super_threshold_indices = self.Data[:,:,0] > 1
        self.Data[super_threshold_indices,0] = 1

        super_threshold_indices2 = self.Data[:,:,0] < 1e-3
        self.Data[super_threshold_indices2,0] = 0
          
            #compensate gaussia blur with multiplication [RIC]
        self.Data[:, :, 0] = gaussian_filter(self.Data[:, :, 0], sigma=0.9, truncate=4.0)
        self.RealWorld=np.zeros((N_totalX,N_totalY))

        # creo una matrice della posizione attuale del robot. la aggiungo alla heatmap,
        #in seconda posizione della terza dimensione del cubo.
        self.Put2Zero()

        #print("state_next ", self.episode_step)
        self.Clean_Walls_and_Pillars()

        self.state_next=self.Data.copy()

    def updateHeatmap2(self,count):
        print("updateHeatmap")
        #if timestep%5==0:
        #for robs in range(self.N_Walkman):

          
            #compensate gaussia blur with multiplication [RIC]
        self.Data[:, :, 0] = gaussian_filter(self.Data[:, :, 0], sigma=0.9, truncate=4.0)

        self.RealWorld=self.RealWorld-SubFerormone

        super_threshold_indices3 = self.RealWorld[:,:] < 1e-3
        self.RealWorld[super_threshold_indices3] = 0 

        super_threshold_indices33 = self.RealWorld[:,:] > 1
        self.RealWorld[super_threshold_indices33] = 1 
        # creo una matrice della posizione attuale del robot. la aggiungo alla heatmap,
        #in seconda posizione della terza dimensione del cubo.
        self.Put2Zero()

        #print("state_next ", self.episode_step)
        self.Clean_Walls_and_Pillars()

        #self.state_next=self.Data


    def Walls_and_Pillars_Matrix(self):
        print("sono Walls_and_Pillars_Matrix")
                

        #N_total=250
        self.Wall_Matrix=np.zeros((N_totalX,N_totalY))
        originalImage = cv2.imread("Termini BeW_1px_1m_100x172.png")
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        myData=np.array(blackAndWhiteImage)
        Convert=np.where(myData==0, 1, myData)
        self.Wall_Matrix=np.where(myData==255,0, Convert) 
    
    def MyReward(self): #modificato
        print("sono nel MyReward")
                
        #print("sono in MyReward",self.episode_step)
        self.reward = self.old_sum - self.new_sum

        #punish no-cleaning actions
        #self.reward=( 1 - (self.episode_step/60) )*self.reward
        if self.reward < 0.001:
            #self.reward = -1
            self.reward = -2.0 #lets try a lower punishment
     

    def Clean_Walls_and_Pillars(self):
        print("sono nel Clean_Walls_and_Pillars")
                
        for i in range(N_totalX-1):
            for j in range(N_totalY-1):
                #print("wall: ", self.Wall_Matrix[i][j])
                if self.Wall_Matrix[i][j] > 0.5:
                    #print("put to zero: ", i, " ", j)
                    self.Data[i, j, 0] = 0
                    self.RealWorld[i, j] = 0

    
    def Build_Walls_and_Pillars(self):
        print("Build_Walls_and_Pillars")
                
        for i in range(N_totalX-1):
            for j in range(N_totalY-1):
                #print("wall: ", self.Wall_Matrix[i][j])
                if self.Wall_Matrix[i][j] > 0.5:
                    #print("put to zero: ", i, " ", j)
                    self.Data[i, j, 1] = 1


    def Put2Zero(self):
        print("sono nel Put2Zero")
                
        super_threshold_indices = self.Data[:,:,0] < 1e-3
        self.Data[super_threshold_indices,0] = 0
        
        super_threshold_indices3 = self.RealWorld[:,:] < 1e-3
        self.RealWorld[super_threshold_indices3] = 0 

        super_threshold_indices33 = self.RealWorld[:,:] > 1
        self.RealWorld[super_threshold_indices33] = 1 

    def step(self,actionX,actionY,DQNAgents,robs,frame_count):
        print("sono nel step")
                
        #self.robs=robs
        self.DQNAgents=DQNAgents
        self.episode_step += 1
        #self.Data[:,:,0]=state_next[:,:,0].copy()
        self.robs=robs
        print("ROBBBB=",self.robs)
        #input("sono in step")
        #if self.robs == 0 and episode_step%2 != 0:
            

        ############################################################
        
        for i in range(0, N_totalX):
            for j in range(0, N_totalY):
                #print("wall: ", self.Wall_Matrix[i][j])
                if self.Wall_Matrix[i][j] == 1:
                    #print("put to zero: ", i, " ", j)
                    self.Data[i,j,0] = 0

        super_threshold_indices2 = self.Data > 1
        self.Data[super_threshold_indices2] = 1
     

        ############################################################


        self.Put2Zero()

        #print("state_next ", self.episode_step)
        self.Clean_Walls_and_Pillars()
        self.old_sum = np.sum(self.Data[:, :, 0])  


        self.actionRobot(actionX,actionY)

         
        self.Put2Zero()
        
        self.new_sum = np.sum(self.Data[:, :, 0]) 
        
        self.MyReward() 
        
        self.done = False

        non_zero_percentage = ( float(np.count_nonzero(self.Data[:, :, 0])) / float(N_totalX * N_totalY) ) * 100.0
            
        if non_zero_percentage < 2.0: #run is solved if 95% of map is clear!
            self.done = True
            print("MAP IS CLEAR at step ", self.episode_step,non_zero_percentage)
        
        #self.state_next=self.Data.copy()
        
        return self.Data,self.reward,self.done
        


    def actionRobot(self,action_after_PredictionX,action_after_PredictionY): #da mettere nel agent (modificato)
        print("sono nel actionRobot")
        print("ROBBBB=",self.robs)
        # inverto le x con le y per fare la trasposta
        MyX=action_after_PredictionX
        MyY=action_after_PredictionY
        #rob_speed = CLEAN_RAY #was 5

        #Gives us 8 total movement options. (0,1,2,3,4,5,6,7)
        Wall_flagX=False
        Wall_flagY=False

        #check the X borders
        if MyX<=(N_totalX-1) and MyX>=0:
            Wall_flagX=True

        #check the Y borders
        if MyY<=(N_totalY-1) and MyY>=0:
            Wall_flagY=True

        #if borders are consistent
        print("MyX,MyY",MyX,MyY)
        if Wall_flagX==True and Wall_flagY==True:
            #check the obstacles
            if self.Wall_Matrix[int(MyX)][int(MyY)]==1:
                self.DQNAgents[self.robs].my_Agent_action_updateX(self.DQNAgents[self.robs].positionRobX[-1])
                self.DQNAgents[self.robs].my_Agent_action_updateY(self.DQNAgents[self.robs].positionRobY[-1])
                print("ROBBBB=",self.robs,"SONO NEL MURO")
            else:
                #update the positions
                self.DQNAgents[self.robs].my_Agent_action_updateX(int(MyX))
                self.DQNAgents[self.robs].my_Agent_action_updateY(int(MyY))
        else:
            self.DQNAgents[self.robs].my_Agent_action_updateX(self.DQNAgents[self.robs].positionRobX[-1])
            self.DQNAgents[self.robs].my_Agent_action_updateY(self.DQNAgents[self.robs].positionRobY[-1])
            print("ROBBBB=",self.robs,"WALLFLAG FALSE")

        self.PositionRobotMatrix()

    def PositionRobotMatrix_Ferormone(self): #modificato Sembra corrispondere a update_positionRobot_Matrix del test26
        print(" sono in PositionRobotMatrix_Ferormone")
        
        #self.RealWorld=np.zeros((N_totalX,N_totalY))
        for i in range(-CLEAN_RAY, +CLEAN_RAY):
            if self.DQNAgents[self.robs].positionRobX[-1] + i >= N_totalX or self.DQNAgents[self.robs].positionRobX[-1] + i < 0:
                continue

            for j in range(-CLEAN_RAY, +CLEAN_RAY):
                if self.DQNAgents[self.robs].positionRobY[-1] + j >= N_totalY or self.DQNAgents[self.robs].positionRobY[-1] + j < 0:
                    continue

                self.RealWorld[self.DQNAgents[self.robs].positionRobX[-1] + i, self.DQNAgents[self.robs].positionRobY[-1] + j] = 1
        #print("self.env.Data",type(self.env.Data),np.shape(self.env.Data))
        #print("self.env.Data",self.env.Data[:,:,1])
        #self.RealWorld[:,:]=self.RobMatrix.copy()
        super_threshold_indices33 = self.RealWorld[:,:] > 1
        self.RealWorld[super_threshold_indices33] = 1  
    

    def PositionRobotMatrix(self): #modificato
        print("sono nel PositionRobotMatrix",self.robs)
        #input("sono in PositionRobotMatrix")        
        #self.RobMatrix=np.zeros((N_totalX,N_totalY))
        #self.Data[:,:,1]=np.zeros((N_totalX,N_totalY))
        #self.Data[:,:,1]=self.Wall_Matrix
        #RIC
        if self.robs==0:
            self.Data[:,:,1]=np.zeros((N_totalX,N_totalY)) #da fare fuori, nel main
        for i in range(-CLEAN_RAY, +CLEAN_RAY):
            if self.DQNAgents[self.robs].positionRobX[-1] + i >= N_totalX or self.DQNAgents[self.robs].positionRobX[-1] + i < 0:
                continue

            for j in range(-CLEAN_RAY, +CLEAN_RAY):
                if self.DQNAgents[self.robs].positionRobY[-1] + j >= N_totalY or self.DQNAgents[self.robs].positionRobY[-1] + j < 0:
                    continue

                self.Data[self.DQNAgents[self.robs].positionRobX[-1] + i, self.DQNAgents[self.robs].positionRobY[-1] + j] = 1
        #print("self.env.Data",type(self.env.Data),np.shape(self.env.Data))
        #print("self.env.Data",self.env.Data[:,:,1])
        #self.Data[:,:,1]=self.RobMatrix.copy()
    

        #self.RobotClean()
        self.clean_robot_position() #RIC

    def clean_robot_position(self): #modificato
        print("sono nel clean_robot_position")
                
        cleaning_ray=CLEAN_RAY #the diameter will than be cleaning_ray*2 + 1
        for i in range(-cleaning_ray, +cleaning_ray):
            if self.DQNAgents[self.robs].positionRobX[-1] + i >= N_totalX or self.DQNAgents[self.robs].positionRobX[-1] + i < 0:
                continue

            for j in range(-cleaning_ray, +cleaning_ray):

                if self.DQNAgents[self.robs].positionRobY[-1] + j >= N_totalY or self.DQNAgents[self.robs].positionRobY[-1] + j < 0:
                    continue

                self.Data[self.DQNAgents[self.robs].positionRobX[-1] + i, self.DQNAgents[self.robs].positionRobY[-1] + j, 0] = 0
        self.PositionRobotMatrix_Ferormone()



    def buildGraph(self):
         
        #OrderAction=list(itertools.permutations([0, 1, 2,3]))
        rnd = np.random
        rnd.seed(91)
        N_subArea=1600
        Side_subArea=int(math.sqrt(N_subArea))
        N_totalX=172
        N_totalY=100
        N_pixel=N_totalX*N_totalY
        K = range(0,4) #aggiunto numero di robot
        N_robots=4
        #n=N_pixel/N_subArea

        #n = 10 #Number of nodes / people to serve / for me 10 bins 10X10
        Q = 1000 #max capacity
        

        self.Data1=np.zeros((N_totalX,N_totalY))
       


        #print("np.shape NewUpgrade",np.shape(NewUpgrade))
        #input("enter")
        self.Data1=self.Data[:,:,0].copy()
        for i in range(0, N_totalX):
            for j in range(0, N_totalY):
                #print("wall: ", self.Wall_Matrix[i][j])
                if self.Wall_Matrix[i][j] == 1:
                    #print("put to zero: ", i, " ", j)
                    self.Data1[i][j] = 0

        super_threshold_indices1 = self.Data1 > 1
        self.Data1[super_threshold_indices1] = 1
        
        super_threshold_indices21 = self.Wall_Matrix > 0
        self.Data1[super_threshold_indices21] = 0

        #super_threshold_indices = NewUpgrade > 1
        #self.NewUpgrade[super_threshold_indices] = 1


    ############################################################
        self.tool2= np.array( [[ 0,  1,  2 ], 
                          [ 3,  4,  5 ],
                          [ 6,  7,  8 ],
                          [ 9,  10, 11 ],
                          [ 12, 13, 14 ] ])
        
        self.tool3= np.array( [[0, 3]])
        #tool3= np.array( [[10, 9],[7,6]])
        #tool3= np.array( [[10, 9],[7,6], [9, 10],[6,7]])
        #tool3= np.array( [[10, 9],[7,6], [9, 10],[6,7],[0,1],[0,3],[0,4],[1,0],[3,0],[4,0]])

    ############################################################
        

        self.NewUpgradeBin=[]
        self.Wall_Bin=[]
        first_limit_bool=False
        second_limit_bool=False

        N_Square_X=int(N_totalX/Side_subArea)
        N_Square_Y=int(N_totalY/Side_subArea)

        Cont_period=Side_subArea-1
        self.StartingPeriodX=[]
        self.StartingPeriodY=[]
        self.EndingPeriodX=[]
        self.EndingPeriodY=[]
        self.Center_SubArea=[]
        
        cont=0    
        for i in range(N_totalX-1):
            
            cont=cont+Cont_period
            
            if i==0:
                cont=i
            
            if first_limit_bool and second_limit_bool:
                break 
            
            cont2=0
            for j in range(N_totalY-1):
                
                cont2=cont2+Cont_period
            
                if j==0:
                    cont2=j

                
                    
                #print("cont",cont)
                #print("cont2",cont2)

                    
                if cont < N_totalX and cont2 < N_totalY:
                    
                    second_limit=cont2+Side_subArea
                    if  cont2+Side_subArea > N_totalY:
                        second_limit=N_totalY
                        second_limit_bool=True
                    
                    first_limit=cont+Side_subArea
                    if  cont+Side_subArea > N_totalX:
                        first_limit=N_totalX
                        first_limit_bool=True
                    #print("first_limit",first_limit)
                    #print("second_limit",second_limit)
                    self.StartingPeriodX.append(cont)
                    self.StartingPeriodY.append(cont2)
                    self.EndingPeriodX.append(first_limit)
                    self.EndingPeriodY.append(second_limit)
                    
                    # devo calcolare il centro, come C=(End-Start)/2 + Start
                    # e salvare i risultati in una lista "distances" con cui
                    # mi calcolo il costo del viaggio (approssimato)
                    StartC = (first_limit-cont)
                    EndC=(second_limit-cont2)
                    
                    Cx=(StartC)/2 + cont
                    Cy=(EndC)/2 + cont2

                    self.Center_SubArea.append([Cx,Cy])
                    #print("Center_SubArea[-1]",Center_SubArea[-1])

                    #print("NewUpgrade",NewUpgrade[cont:first_limit,cont2:second_limit])
                    self.NewUpgradeBin.append(self.Data1[cont:first_limit,cont2:second_limit].copy())
                    self.Wall_Bin.append(self.Wall_Matrix[cont:first_limit,cont2:second_limit].copy())
        
        #print("len(NewUpgradeBin)",len(NewUpgradeBin))  




        
        self.distances=np.zeros((len(self.NewUpgradeBin),len(self.NewUpgradeBin) ))

        for i in range(len(self.NewUpgradeBin)):
            
            for j in range(len(self.NewUpgradeBin) ):
                Onedistance = np.hypot(self.Center_SubArea[i][0] - self.Center_SubArea[j][0], self.Center_SubArea[i][1] - self.Center_SubArea[j][1])
                print("distances[i][j]",Onedistance)
                self.distances[i][j]=Onedistance
        
        
        self.List_Distance_Between_RobsAndNodes=[]
        self.List_Distance_Between_RobsAndNodes_forEveryRobs=[]
        for robs in range(N_robots ):
            for i in range(len(self.NewUpgradeBin)):
                
                Onedistance = np.hypot(self.DQNAgents[robs].positionRobX[-1] - self.Center_SubArea[i][0], self.DQNAgents[robs].positionRobY[-1] -self.Center_SubArea[i][1])
            
                #print("distances[i][j]",Onedistance)
            
                self.List_Distance_Between_RobsAndNodes.append(Onedistance)
            self.List_Distance_Between_RobsAndNodes_forEveryRobs.append(self.List_Distance_Between_RobsAndNodes)
        
        self.Initialize_q_cPiu_cMeno_distances_Starting()


    def Initialize_q_cPiu_cMeno_distances_Starting(self):



        #####################################################################################################
        # creazione di Starting
        self.Starting=[] # ci sono le posizioni iniziali in termini di nodo

        for i in range(N_robots):
            yPerNode=self.DQNAgents[i].positionRobX[-1]
            xPerNode=self.DQNAgents[i].positionRobY[-1]
            j=0
            boolRange=True
            while boolRange and j<len(self.StartingPeriodX):
                #print("xPerNode",xPerNode)
                #print("yPerNode",yPerNode)
                #print("StartingPeriodX[j]",StartingPeriodX[j])
                #print("EndingPeriodX[j]",EndingPeriodX[j])
                #print("StartingPeriodY[j]",StartingPeriodY[j])
                #print("EndingPeriodY[j]",EndingPeriodY[j])
                #print ("j",j)
                if yPerNode>=self.StartingPeriodX[j] and yPerNode<=self.EndingPeriodX[j] and xPerNode>=self.StartingPeriodY[j] and xPerNode<=self.EndingPeriodY[j]:
                    self.Starting.append(j)
                    boolRange=False
                    break
                j=j+1


        #######################################################################################################



        #NewUpgradeBin2.reverse()
        ######################################################################################    
        n=len(self.NewUpgradeBin)-1
        N = [i for i in range(1, n+1)] 
        V = [0] + N
        ##########################################################################################


        ValueNode=np.zeros(len(self.NewUpgradeBin))
        #print("ValueNode dim",np.shape(ValueNode))
        #print("NewUpgradeBin[k]",np.shape(NewUpgradeBin[8]))
        #print("NewUpgradeBin",np.shape(NewUpgradeBin))
        #print("Side_subArea",Side_subArea) 
        
        #Values of my q
        for k in range(len(self.NewUpgradeBin)):
            dimBin=np.shape(self.NewUpgradeBin[k])
            print("dimBin 0",dimBin[0])
            print("dimBin 1",dimBin[1])
            for i in range(dimBin[0]):
                for j in range(dimBin[1]):
                    #print("k",k)
                    #print("i",i)
                    #print("j",j)
                    if self.Wall_Bin[k][i][j]==0:
                        print("NewUpgradeBin[k][i][j]",self.NewUpgradeBin[k][i][j])
                        ValueNode[k]=ValueNode[k]+self.NewUpgradeBin[k][i][j]/1000
                        #print(ValueNode[k])
        #ValueNode=[0.6, 0.1, 0.2, 0.5, 0.6, 0.8, 0.1, 0.7, 0.4, 0.7, 0.7, 0.3, 0.4,0.2, 0.5]
        #print("ValueNode",ValueNode)
        #return
        #print("N",N)
        #ValueNode=np.flip(ValueNode,0)
        #q = {i:0.1 if i in tool3  else ValueNode[i] for i in N}
        
        
        #print("q",q2)

        #############################################################################################
           
        self.q2={i: 0 if i in self.tool3  else ValueNode[i] for i in V }
        
        A = [(i, j) for i in V for j in V ]
       
        #Starting=[ 10, 5, 6, 2]
        
        #for k in range(len(Starting)):
        
        #    A.append((0,Starting[k],k))
        
        #devo calcolarmi tutte le distanze tra i nodi

        self.cPiu = {(i, j): self.q2[j]  for i, j in A }
        
        self.cPiu[(0, 1) ]=0
        self.cPiu[(1, 0) ]=0
        self.cPiu[(0, 3) ]=0
        self.cPiu[(3, 0) ]=0
        self.cPiu[(3, 1) ]=0
        self.cPiu[(1, 3) ]=0
        self.cPiu[(3, 1) ]=0

        self.cMeno= {(i, j): self.distances[i][j]  for i, j in A }
        self.cMeno[(0, 1) ]=10000
        self.cMeno[(1, 0) ]=10000
        self.cMeno[(0, 3) ]=10000
        self.cMeno[(3, 0) ]=10000
        self.cMeno[(3, 1) ]=10000
        self.cMeno[(1, 3) ]=10000
        self.cMeno[(3, 1) ]=10000
        
        for i in range(len(self.NewUpgradeBin)):
            
            self.cMeno[(i, i) ]=10000
            self.cPiu[(i, i) ]=0

        self.Number_bugs=[]

        for n in range(len(self.NewUpgradeBin)):
            cont=0
            for k in range(N_robots):
            
                if self.Starting[k]==n:
                    cont=cont+1
            self.Number_bugs.append(cont)

        #return Number_bugs, cPiu,cMeno NewUpgradeBin

    def myFood_Input_Fuzzy1(self):

        self.Initialize_q_cPiu_cMeno_distances_Starting()
         
        List_MyChoice=[]

        for k in range(N_robots):

        
            # ho bisogno di 2 conti con Fuzzy.
            # il primo è la scelta della sub-area.
            # una volta scelta l'area di interesse, devo azzerare il valore della relativa area, per evitare che
            #tutti i robot scelgano la stessa.
            List_forbidden_Node = [0,1,3]
            List_Next_Node = []
            for n in range(len(self.NewUpgradeBin)):

                if n in List_forbidden_Node:
                    List_Next_Node.append(-1)
                    self.q2[n]=0

                else:            
                    
                    Next_node_Evaluation=self.type2Fuzzy_Control1(self.q2[n],self.cMeno[(self.Starting[k], n) ],self.cPiu[(self.Starting[k], n) ], self.Number_bugs[n])
                    List_Next_Node.append(Next_node_Evaluation)
            
                    

            max_item = max(List_Next_Node)
                
            print("max_item",max_item)

            MychoiceCurrent=List_Next_Node.index(max_item) 
            

            print("MychoiceCurrent",MychoiceCurrent)
            List_forbidden_Node.append(MychoiceCurrent)
            print("List_forbidden_Node",List_forbidden_Node)
            #List_Next_Node.sort(reverse=True)
            List_MyChoice.append(MychoiceCurrent)
            #print("Mychoice",Mychoice)
            for y in range(N_robots):
                self.cMeno[(self.Starting[y], MychoiceCurrent) ]=10000
                self.cPiu[(self.Starting[y], MychoiceCurrent) ]=0
            
            for nn in range(len(self.NewUpgradeBin)):
                self.cMeno[(nn, MychoiceCurrent) ]=10000
                self.cPiu[(nn, MychoiceCurrent) ]=0            

            
        
        return List_MyChoice, self.Center_SubArea
           




    def type2Fuzzy_Control1(self,q2,cMeno,cPiu, Number_bugs):
        
        
        x1=cMeno/1
        #x2=cPiu*1
        x2=q2
        x3=Number_bugs

        #x3=MyGrid_priority
        print("x1",x1,"x2",x2,"x3",x3)
        #input("enter")
        #Defining the domain of the output
        #domain = linspace(1., 0.15, 10)
        domain = linspace(0., 1., 100000)

        # Defining the domain of the input variable Goal_manhattan_Grid,
        domain1 = linspace(0., 150., 100000)

        # Defining the domain of the input variable d_Euclidean,
        domain2 = linspace(0., 450., 100000)

        # Defining the domain of the input variable MyGrid priority.
        domain3 = linspace(0., 4., 100000)

        #The params input for IT2FS_Gaussian_UncertStd function, is a list consisting of the mean, the standard deviation center, the standard deviation spread, and the height of the set

        # Defining the Small set for the input variable Goal_manhattan_Grid.
        #R_IT2FS_Gaussian_UncertStd
        Small1 = IT2FS_Gaussian_UncertStd(domain1, [0., 8., 5., 1.])

        # Defining the Small set for the input variable d_Euclidean.
        #Small2 = IT2FS_Gaussian_UncertStd(domain, [0.4, 0.15, 0.05, 1.])

        # Defining the Small set for the input variable MyGrid priority.
        Small2 = IT2FS_Gaussian_UncertStd(domain2, [0., 45., 15., 1.])

        Small3 = IT2FS_Gaussian_UncertStd(domain3, [0., 0.15, 0.05, 1.])


        # Defining the Medium set for the input variable Goal_manhattan_Grid.
        Medium1 = IT2FS_Gaussian_UncertStd(domain1, [75., 8., 5., 1.])

        # Defining the Medium set for the input variable d_Euclidean.
        #Medium2 = IT2FS_Gaussian_UncertStd(domain, [0.5, 0.15, 0.05, 1.])

        # Defining the Medium set for the input variable MyGrid priority.
        Medium2 = IT2FS_Gaussian_UncertStd(domain2, [225., 45., 15., 1.])

        Medium3 = IT2FS_Gaussian_UncertStd(domain3, [2., 0.15, 0.05, 1.])

        # Defining the Large set for the input variable Goal_manhattan_Grid.
        Large1 = IT2FS_Gaussian_UncertStd(domain1, [150., 8., 5., 1.])

        # Defining the Large set for the input variable d_Euclidean.
        #Large2 = IT2FS_Gaussian_UncertStd(domain, [0.6, 0.15, 0.05, 1.])

        # Defining the Large set for the input variable MyGrid priority.
        Large2 = IT2FS_Gaussian_UncertStd(domain2, [450., 45., 15., 1.])

        Large3 = IT2FS_Gaussian_UncertStd(domain3, [4., 0.15, 0.05, 1.])

        # Plotting the sets defined for the input variable x1.
        #IT2FS_plot(Small1, Medium1, Large1, legends=["Small", "Medium", "large"])

        # Plotting the sets defined for the input variable x2.
        #IT2FS_plot(Small2, Medium2, Large2, legends=["Small", "Medium", "large"])


        # Plotting the sets defined for the input variable x2.
        #IT2FS_plot(Small3, Medium3, Large3, legends=["Small", "Medium", "large"])

        # Defining the mamdani interval type 2 fuzzy logic system
        myIT2FLS = IT2Mamdani(min_t_norm, max_s_norm, method="Centroid", algorithm="KM")
        #myIT2FLS = Mamdani(min_t_norm, max_s_norm)

        # Adding the input variables to the myIT2FLS
        myIT2FLS.add_input_variable("x1")
        myIT2FLS.add_input_variable("x2")
        myIT2FLS.add_input_variable("x3")
        #myIT2FLS.add_input_variable("x4")
        # Adding the output variables to the myIT2FLS


        LB = IT2FS_Gaussian_UncertStd(domain, [0., 0.02, 0.01, 1.])
        LM = IT2FS_Gaussian_UncertStd(domain, [0.11, 0.02, 0.01, 1.])
        LA = IT2FS_Gaussian_UncertStd(domain, [0.22, 0.02, 0.01, 1.])

        MB = IT2FS_Gaussian_UncertStd(domain, [0.33, 0.02, 0.01, 1.])
        MM = IT2FS_Gaussian_UncertStd(domain, [0.44, 0.02, 0.01, 1.])
        MA = IT2FS_Gaussian_UncertStd(domain, [0.55, 0.02, 0.01, 1.])

        HB = IT2FS_Gaussian_UncertStd(domain, [0.66, 0.02, 0.01, 1.])
        HM = IT2FS_Gaussian_UncertStd(domain, [0.77, 0.02, 0.01, 1.])
        HA = IT2FS_Gaussian_UncertStd(domain, [0.88, 0.02, 0.01, 1.])
        HAA = IT2FS_Gaussian_UncertStd(domain, [1., 0.02, 0.01, 1.])

        #IT2FS_plot(NB, NM, ZZ, PM, PB, legends=["Negative Big", "Negative Medium", 
        #                                       "Zero", "Positive Medium", 
        #                                       "Positive Big"], filename="delay_pid_output_sets")

        '''
        # The Small set is defined as a Guassian IT2FS with uncertain standard deviation 
        # value. The mean, the standard deviation center, the standard deviation spread, 
        # and the height of the set are set to 0., 0.15, 0.1, and 1., respectively.
        Small = L_IT2FS_Gaussian_UncertStd(domain, [0, 0.15, 0.05, 1.])

        # The Medium set is defined as a Guassian IT2FS with uncertain standard deviation 
        # value. The mean, the standard deviation center, the standard deviation spread, 
        # and the height of the set are set to 0.5, 0.15, 0.1, and 1., respectively.
        Medium = L_IT2FS_Gaussian_UncertStd(domain, [0.5, 0.15, 0.05, 1.])

        # The Large set is defined as a Guassian IT2FS with uncertain standard deviation 
        # value. The mean, the standard deviation center, the standard deviation spread, 
        # and the height of the set are set to 1., 0.15, 0.1, and 1., respectively.
        Large = L_IT2FS_Gaussian_UncertStd(domain, [1., 0.15, 0.05, 1.])
        '''
        #x1=cMeno/1 distanza
        #x2=cPiu*1 quantità di sporco
        #x3=Number_bugs

        myIT2FLS.add_output_variable("y1")  


                           # distance     # numbero of bugs   # priorities 
        myIT2FLS.add_rule([("x1", Small1), ("x3", Small3),("x2", Small2)], [("y1",LA )]) 

        myIT2FLS.add_rule([("x1", Medium1), ("x3", Small3),("x2", Small2)], [("y1", LM)]) 

        myIT2FLS.add_rule([("x1", Large1), ("x3", Medium3),("x2", Small2)], [("y1", LB)])

        
        myIT2FLS.add_rule([("x1", Small1), ("x3", Medium3),("x2", Medium2)], [("y1", MA)])

        myIT2FLS.add_rule([("x1", Medium1), ("x3", Large3),("x2", Medium2)], [("y1", MM)]) 

        myIT2FLS.add_rule([("x1", Large1), ("x3", Large3),("x2", Medium2)], [("y1",MB )]) 


        myIT2FLS.add_rule([("x1", Small1), ("x3", Small3),("x2", Large2)], [("y1",HAA )])

        myIT2FLS.add_rule([("x1", Medium1), ("x3", Small3),("x2", Large2)], [("y1", HM)])

        myIT2FLS.add_rule([("x1", Large1), ("x3", Medium3),("x2", Large2)], [("y1", HB)])

      



        myIT2FLS.add_rule([("x1", Small1), ("x3", Medium3),("x2", Small2)], [("y1", LA)])

        myIT2FLS.add_rule([("x1", Medium1), ("x3",Large3),("x2", Small2)], [("y1", LM)]) 

        myIT2FLS.add_rule([("x1", Large1), ("x3", Large3),("x2", Small2)], [("y1",LB )]) 


        myIT2FLS.add_rule([("x1", Small1), ("x3", Small3),("x2", Medium2)], [("y1", MA)]) 

        myIT2FLS.add_rule([("x1", Medium1), ("x3", Small3),("x2", Medium2)], [("y1", MM)]) 

        myIT2FLS.add_rule([("x1", Large1), ("x3", Medium3),("x2", Medium2)], [("y1",MB )])

        
        myIT2FLS.add_rule([("x1", Small1), ("x3", Medium3),("x2", Large2)], [("y1", HA)]) 

        myIT2FLS.add_rule([("x1", Medium1), ("x3", Large3),("x2", Large2)], [("y1", HM)]) 

        myIT2FLS.add_rule([("x1", Large1), ("x3", Large3),("x2", Large2)], [("y1",HB )]) 

        myIT2FLS.add_rule([("x1", Large1), ("x3", Large3),("x2", Large2)], [("y1",HB )]) 


        c, TR = myIT2FLS.evaluate({"x1":x1, "x2":x2,"x3":x3})
        #c, TR = myIT2FLS.evaluate({"x1":x3, "x2":0.745})
        print("TR",TR)
        print("shape TR",np.shape(TR))
        print("c",c)
        o = TR["y1"]
        print("o",o)
        o1 = (TR["y1"][0] + TR["y1"][1] ) / 2
        print("o1",o1)
            

            #input("enter")
            

        
        return o1
        #input("enter")

    # p=myFood_Input_Fuzzy2(List_MyDestinationNodeChoice,contPriorities,contFerormone,DQNAgents,env,inx2,iny2,robs)




def Walls_and_Pillars_Matrix():
    print("sono Walls_and_Pillars_Matrix")
            

    #N_total=250
    Wall_Matrix=np.zeros((N_totalX,N_totalY))
    originalImage = cv2.imread("Termini BeW_1px_1m_100x172.png")
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    myData=np.array(blackAndWhiteImage)
    Convert=np.where(myData==0, 1, myData)
    Wall_Matrix=np.where(myData==255,0, Convert) 

    MyFrameCSV="Wall_Matrix.csv"
    pd.DataFrame(Wall_Matrix).to_csv(MyFrameCSV)   


    return Wall_Matrix


def import_data():
    
    print("sono nel import_data_old_Week")
                
    # Frame list contiene i numeri associati ai frame di ciascuno dei giorni scelti
    #Scelgo 2 orari da confrontare
    myFrame1='Frame_Second_Half_d_0_h_11.csv'
    
    #usecols=[1,172],userows=[1,100],skiprows=1,usecols=[1:100]
    MyState=np.zeros((N_totalX,N_totalY))
    MyState=pd.read_csv(myFrame1,header=0,index_col=0)

    super_threshold_indices11 = MyState == 0.9
    MyState[super_threshold_indices11] = 0.75

    super_threshold_indices22 = MyState == 0.8
    MyState[super_threshold_indices22] = 0.50

    super_threshold_indices33 = MyState == 0.7
    MyState[super_threshold_indices33] = 0.25
    #MyPlot2(MyState)
    #print("MyState",np.shape(MyState))
    return MyState

def MyPlot(Wall_Matrix,state,chebyshevGrid):
    
    #env.Data=state

    fig = plt.figure(1)
    fig.clf()

    ax = fig.subplots(nrows=2, ncols=2) # , figsize=(16, 4))

    # ax1 = fig.add_subplot(221)
    extent = (0, N_totalY, 0, N_totalX)
    ax[0, 0].imshow(state, cmap=plt.cm.hot, origin='upper', extent=extent)
    #ax[0, 0].imshow(state[:, :], cmap='cubehelix',alpha=0.5, origin='upper', extent=extent)
    ax[0, 0].imshow(Wall_Matrix, cmap='copper', alpha=0.5, origin='upper', extent=extent)

    ax[0, 0].set_title('State')
    ax[0, 0].set_xlabel('x-cord')
    ax[0, 0].set_ylabel('y-cord')

    # ax2 = fig.add_subplot(222)
    #cmap='cubehelix'
    #ax[0, 1].imshow(env.Data[:, :, 0], cmap=plt.cm.hot, origin='upper', extent=extent)
    ax[0, 1].imshow(chebyshevGrid[:, :], cmap='gray', origin='upper', extent=extent)
    #ax[0, 1].imshow(env.Wall_Matrix, cmap='copper', alpha=0.5, origin='upper', extent=extent)

    ax[0, 1].set_title('Robots Positions')
    ax[0, 1].set_xlabel('x-cord')
    ax[0, 1].set_ylabel('y-cord')

    plt.show()
    #plt.pause(0.0001)





def get_motion_model():
    # dx, dy
    motion = [[1, 0],
              [0, 1],
              [-1, 0],
              [0, -1],
              [-1, -1],
              [-1, 1],
              [1, -1],
              [1, 1]]

    return motion

def Priority_robot_position(px,py,robs,env):

    cleaning_ray=CLEAN_RAY #the diameter will than be cleaning_ray*2 + 1
    
    contPriorities=0
    contFerormone=0
    for i in range(-CLEAN_RAY, +CLEAN_RAY):
        if px + i >= N_totalX or px + i < 0:
            continue

        for j in range(-CLEAN_RAY, +CLEAN_RAY):
            if py + j >= N_totalY or py + j < 0:
                continue

            contPriorities=contPriorities+env.Data[px + i, py + j,0]
            contFerormone=contFerormone+env.RealWorld[px + i, py + j]
    #contPriorities=env.Data[px, py,0]
    return contPriorities,contFerormone

#rx,ry = build_Greedy_path(DQNAgents,robs,env.Wall_Matrix,env,List_MyDestinationNodeChoice) 
                
def build_Greedy_path(DQNAgents,robs,Wall_Matrix,env,List_MyDestinationNodeChoice,Center_SubArea):    
    # calc potential field
    print("sono in build_Greedy_path, robs",robs)    
    # search path
    ix=DQNAgents[robs].positionRobX[-1]
    iy=DQNAgents[robs].positionRobY[-1]
    print("ix",ix)
    print("iy",iy)
    minp = -1
    
    
    minix, miniy = -1, -1
    motion = get_motion_model()
    contMinpMin=0
    
    
    for i in range(len(motion)):
        
        inx = int(ix + motion[i][0]*robspeed)
        iny = int(iy + motion[i][1]*robspeed)
        print("inx",inx)
        print("iny",iny)

        
        
        
        if inx >= N_totalX or iny >= N_totalY or inx<0 or iny<0 or Wall_Matrix[inx,iny]>0.5 :
            print("sono nel muro")
            p = -1  # outside area
        
        else:
            inx2 = int(ix + motion[i][0]*(robspeed+1))
            iny2 = int(iy + motion[i][1]*(robspeed+1))
            
            if inx2 >= N_totalX or iny2 >= N_totalY or inx2<0 or iny2<0 or Wall_Matrix[inx2,iny2]>0.5 :
                p = -1
               
            else:
            
                #return contPriorities,contFerormone
                contPriorities,contFerormone = Priority_robot_position(inx2,iny2,robs,env)
                
                #def myFood_Input_Fuzzy2(List_MyChoice,current_dark_choice, current_feromone_choice,DQNAgents,env,inx,iny,robs,Center_SubArea):

                p=myFood_Input_Fuzzy2(List_MyDestinationNodeChoice,contPriorities,contFerormone,DQNAgents,env,inx2,iny2,robs,Center_SubArea)



            print("sono prima dell'if minp<p minp, p,motion[i][0],motion[i][1],robs", minp, p,motion[i][0],motion[i][1],robs)
            #input("enter")
            if minp < p:
                print("sono nell'if minp<p,p,motion[i][0],motion[i][1],robs", minp, p,motion[i][0],motion[i][1],robs)
                minp = p
                minix = inx
                miniy = iny
                #input("perché sempre giu")

    ix = minix
    iy = miniy
    print("ix",ix)
    print("iy",iy)
    #input("enter 1")
    print("Goal!!")
    DQNAgents[robs].positionRobX.append(ix)
    DQNAgents[robs].positionRobY.append(iy)



    return ix,iy



 



def draw_heatmap(data,MyObstacleMap):

   
    #data = np.array(data)
    #plt.pcolor(data, vmax=100.0, cmap=plt.cm.hot)
    plt.figure(3) 
    #extent = (0,N_totalX,0, N_totalY)   
    plt.imshow(data,cmap=plt.cm.hot,origin='upper')
    plt.imshow(MyObstacleMap,cmap='copper', alpha=0.5,origin='upper')
    plt.draw()
    plt.pause(0.0001)
    #cmap=plt.cm.Blues            

def main():


    signal.signal(signal.SIGQUIT, changeVisualization)
    max_steps_per_episode = STEPS_PER_EPISODE
        #for PLOT
    episode_rewards_log = np.zeros((100000000, 3))
    robot_rewards_log = np.zeros((STEPS_PER_EPISODE, N_robots+1))
    robot_rewards_log[:,0] = range(0, max_steps_per_episode)
    color_map = plt.cm.rainbow
    color_norm = matplotlib.colors.Normalize(vmin=1.5, vmax=4.5)
    stored_execution_time = time.time()
    max_steps_per_episode = STEPS_PER_EPISODE
    seed = 42
    cumulative_rewards_log = np.zeros((10000000, 2))
    single_robot_rewards_log = np.zeros((10000000, N_robots))
    cumulative_step = 0
    
 #   single_robot_rewards = np.zeros((4,1))
    episode_count = 0
    frame_count = 0
    stored_execution_time = time.time()
    running_reward=0
    episode_reward=0
    DoneRunning=False
    
    #episode_not_zero_percentage_history = [] # only to evaluate test;
    episode_PercentageSomma_history = [] # only to evaluate test;
    episode_PercentageSomma_history3 = []
    TENepisode_PercentageSomma_history_max = [] # only to evaluate test;
    TENepisode_PercentageSomma_history_min = [] # only to evaluate test;
    TENepisode_PercentageSomma_history_mean = [] # only to evaluate test;
    TENepisode_PercentageSomma_history_Var=[] # new!
    TENepisode_PercentageSomma_history_Var3=[] # new new!
    TENepisode_PercentageSomma_history_mean3 = []
    episode_reward_history = []

    ######################################NeW ########################
    #primo blocco, tutto ok
    TENrunning_rewardMIN=[]
    TENrunning_rewardMean=[]
    TENrunning_rewardMAX=[]
    TENrunning_rewardSTD=[]
    episode_reward_history = []
    running_Variance=[]
    StepReward_history=[]
    TEN_reward_Somma=[]
    #N_upp=0
    ################################################################## End New
    signal.signal(signal.SIGQUIT, changeVisualization)

    
    MyEpTest=0; 
    while MyEpTest<1: 



        MyEpTest=MyEpTest+1
        
        #env.reset(robs,StartX,StartY)
        #episode_not_zero_percentage_history.clear()
        episode_PercentageSomma_history.clear()
        episode_PercentageSomma_history3.clear()
        #del episode_not_zero_percentage_history[:1]
        del episode_PercentageSomma_history[:1]
        del episode_PercentageSomma_history3[:1]

    
    #while True:  # Run until solved
        
        DQNAgents = []
        
        #n_node=0
        stored_execution_time = time.time()
        for Rango in range(N_robots):
            DQNAgents.append(DQNAgent())
        
        env = BlobEnv(DQNAgents)

        agent_position=[]
        for robs in range(N_robots):
        
            #env.reset(robs,StartX(robs,n_node),StartY(robs,n_node)) #è da fare per ogni robot?
            env.reset(robs,StartListX[robs][DQNAgents[robs].current_n_node],StartListY[robs][DQNAgents[robs].current_n_node]) #è da fare per ogni robot?
            #input("dopo reset")
            #env.update_positionOfRobot_Matrix(agent_position[0],agent_position[1],robs)
            #print("main init pos received")
        
        #state = np.array(env.Data)   ###########################################
        episode_reward = 0        
        
        #EvenOdd==True
        stored_execution_time = time.time()
        ############################################## forth new
        episode_PercentageSomma_history.clear()
        episode_PercentageSomma_history3.clear()
        StepReward_history.clear()
        ############################################## end 
        
        countUpdatePeriod=0
        

   
        for timestep in range(1, max_steps_per_episode+1):
            #print("prima if countUpdatePeriod",countUpdatePeriod)  
             #input("enter")
            #if (timestep%update_period==0) and (countUpdatePeriod<=63):

                #print("dopo if countUpdatePeriod",countUpdatePeriod)    
                #input("enter")
            timestep1=timestep-1
            if ((timestep1==countUpdatePeriod*update_period) and (countUpdatePeriod<15) ):
           
                env.updateHeatmap(countUpdatePeriod)
                env.buildGraph()
                print("timestep,countUpdatePeriod",timestep,countUpdatePeriod)
                #input("dopo updateHeatmap")
                countUpdatePeriod+=1
                
                # invio l'env generale aggiornato ai processi 
                #dopo l'applicazione della gaussiana
            
            #print("terza comunicazione:invio l'env.Data aggiornato al processo",type(env.Data),np.shape(env.Data))
            
            else:
                env.updateHeatmap2(countUpdatePeriod)
            
            List_MyDestinationNodeChoice,env.Center_SubArea= env.myFood_Input_Fuzzy1()

            #overall_state = env.Data.copy()
            #SommaBefore=np.sum(env.Data[:,:,0])
            StepReward=0
            
            

            for robs in range(N_robots):
                #overall_state[:,:,1]=env.PositionRobotMatrix2(DQNAgents,robs)
                # aggiungo le posizioni a overall state e a env.Data, e pulisco quelle posizioni,
                # che lo sporco appena aggiunto ha coperto (forse da commentare)
                
                #rx,ry,doneNode =build_Boustrophedon_path(DQNAgents,robs,env.Wall_Matrix)
                #rx,ry,doneNode=build_Spiral_path(DQNAgents,robs, env.Wall_Matrix)
                
                rx,ry = build_Greedy_path(env.DQNAgents,robs,env.Wall_Matrix,env,List_MyDestinationNodeChoice,env.Center_SubArea) 
                #StartPosition=np.array([rx,ry])
                #GoalPosition=np.array([EndX(robs,n_node),EndY(robs,n_node)])
                

                state_next, reward, done = env.step(rx,ry,env.DQNAgents,robs,frame_count) # devo modificare, aggiornare lo stato dell'env.
                robot_rewards_log[timestep-1, robs+1] = reward       
                #overall_state[:,:,1] = overall_state[:,:,1] + state_next[:, :, 1]
                #overall_state[:, :, 0] = state_next[:,:,0]
                env.Data=state_next.copy()
                episode_reward += reward
                StepReward = StepReward+reward
                

            SommaTot=np.sum(env.Data[:,:,0])

            
            StepReward_history.append(StepReward) ## NEW
            #MaxRisk=N_Walkman+env.first_sum
            MaxRisk2=float(N_totalX * N_totalY)-float(np.count_nonzero(env.Wall_Matrix))
            #print("SommaTot,MaxRisk",SommaTot,MaxRisk2)
            PercentageSomma=((MaxRisk2-SommaTot)/MaxRisk2)*100
            
            ####################### I define a new percentage of cleaning for a section of area

            SommaTot3=np.sum(env.Data[0:171,79:98,0])
            #print("Dimensioni di DATA0",np.shape(env.Data[:,:,0]),env.Wall_Matrix[171,78])
            MaxRisk3=3268 #172*19
            PercentageSomma3=((MaxRisk3-SommaTot3)/MaxRisk3)*100

            ###################################################################################


            episode_PercentageSomma_history.append(PercentageSomma)
            episode_PercentageSomma_history3.append(PercentageSomma3)
            ###############################################################################11th block

            if ((timestep)%update_period==0) and (countUpdatePeriod<15) or (timestep==60):
            

                with open("ReportDatiMerakiColonyNew"+str(N_robots)+str(N_Walkman)+".txt", "a") as f:
                    print(" ",PercentageSomma," ",StepReward," ",PercentageSomma3, file=f)
                #print(" ciao",PercentageSomma," ciao",StepReward)
            ###############################################################################end
            
            if ((timestep1==(countUpdatePeriod-1)*update_period) and (countUpdatePeriod-1<15) ):

                print("timestep Metto sporco,Forza",timestep,countUpdatePeriod,PercentageSomma)
      
            if TESTING:
                # plt.clf()
                fig = plt.figure(2)
                fig.clf()

                ax = fig.subplots(nrows=2, ncols=2)  # , figsize=(16, 4))

                # ax1 = fig.add_subplot(221)
                extent = (0, N_totalY, 0, N_totalX)
                ax[0, 0].imshow(env.Data[:, :, 0], cmap=plt.cm.hot, origin='upper', extent=extent)

                ax[0, 0].set_title('State')
                ax[0, 0].set_xlabel('x-cord')
                ax[0, 0].set_ylabel('y-cord')

                # ax2 = fig.add_subplot(222)
                #cmap='cubehelix'
                ax[0, 1].imshow(env.RealWorld, cmap='copper', origin='upper', extent=extent)
                ax[0, 1].imshow(env.Data[:, :, 1], cmap='gray', alpha=0.5, origin='upper', extent=extent)
                #ax[0, 1].imshow(env.Wall_Matrix, cmap='copper', alpha=0.5, origin='upper', extent=extent)

                ax[0, 1].set_title('Robots Positions')
                ax[0, 1].set_xlabel('x-cord')
                ax[0, 1].set_ylabel('y-cord')

                # ax3 = fig.add_subplot(223)
                for k in range(N_robots):
                    ax[1, 0].plot(robot_rewards_log[0:timestep, 0],
                                  robot_rewards_log[0:timestep, k + 1], color=color_map(color_norm(k)))

                ax[1, 0].set_title('Robot-specific rewards')
                ax[1, 0].set_xlabel('runs')
                ax[1, 0].set_ylabel('rewards')

                plt.draw()
                plt.pause(0.0001)
            # PLOT Figure 2 - end
        
        

        episode_reward_history.append(episode_reward)
        
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
            del episode_PercentageSomma_history[:1]

        running_reward = np.mean(episode_reward_history)

        reward_Somma=np.sum(StepReward_history)
        running_Somma=np.mean(episode_PercentageSomma_history)
        running_Somma3=np.mean(episode_PercentageSomma_history3)
        max_Somma_history=np.max(episode_PercentageSomma_history)
        min_Somma_history=np.min(episode_PercentageSomma_history)
        
        ######################################################fifth block 
        #testato tutto ok
        running_Variance=np.std(episode_PercentageSomma_history) # Standard deviation della percentuale di pulizia
        running_Variance3=np.std(episode_PercentageSomma_history3) # Standard deviation della percentuale di pulizia
        running_rewardMean = np.mean(StepReward_history) # media del reward
        running_rewardSTD = np.std(StepReward_history) # Standard deviation del reward
        running_rewardMIN=np.min(StepReward_history)#min reward
        running_rewardMax=np.max(StepReward_history)#max reward

        ###################################################################

        TEN_reward_Somma.append(reward_Somma)
        #max_not_zero_perc=np.max(episode_not_zero_percentage_history)
        #min_not_zero_perc=np.min(episode_not_zero_percentage_history)
        TENepisode_PercentageSomma_history_max.append(max_Somma_history)
        TENepisode_PercentageSomma_history_min.append(min_Somma_history)
        TENepisode_PercentageSomma_history_mean.append(running_Somma)
        TENepisode_PercentageSomma_history_mean3.append(running_Somma3)
        ################################################################# sixth block
        TENepisode_PercentageSomma_history_Var.append(running_Variance)
        TENepisode_PercentageSomma_history_Var3.append(running_Variance3)
        TENrunning_rewardMean.append(running_rewardMean)
        TENrunning_rewardSTD.append(running_rewardSTD)
        TENrunning_rewardMIN.append(running_rewardMIN)
        TENrunning_rewardMAX.append(running_rewardMax)


        #############################################################################

        print("N_Walkman",N_Walkman,"episode: ", episode_count, " reward: ", episode_reward, " elapsed: ","PercentagePulizia",running_Somma,"timestep",timestep,
              time.time() - stored_execution_time,"min_Somma_history",min_Somma_history,"max_Somma_history",max_Somma_history)
        #print("timestep: ", timestep, ", robot: ", robs, ", history: ", len(rewards_history))
        ######################################################################### 9th block
        print("N_robots",N_robots,"N_Walkman",N_Walkman,"episode: ", episode_count, " reward: ", episode_reward, " elapsed: ","PercentagePulizia",running_Somma,"timestep",timestep, time.time() - stored_execution_time,"min_Somma_history",min_Somma_history,"max_Somma_history",max_Somma_history,"running_Variance",running_Variance, "running_rewardMean",running_rewardMean,"running_rewardSTD",running_rewardSTD,"running_rewardMIN",running_rewardMIN,"running_rewardMax",running_rewardMax)
        
        with open("ReportDatiMerakiColonyNew"+str(N_robots)+str(N_Walkman)+".txt", "a") as f:
            print("per ogni Eposodio: N_robots",N_robots,"N_Walkman",N_Walkman,"episode: ", episode_count, " reward: ", episode_reward, " elapsed: ","PercentagePulizia",running_Somma,"timestep",timestep, time.time() - stored_execution_time,"min_Somma_history",min_Somma_history,"max_Somma_history",max_Somma_history,"running_Variance",running_Variance, "running_rewardMean",running_rewardMean,"running_rewardSTD",running_rewardSTD,"running_rewardMIN",running_rewardMIN,"running_rewardMax",running_rewardMax,file=f)
        
        #########################################################################end
        stored_execution_time = time.time()
        stored_execution_time = time.time()

        episode_rewards_log[episode_count, 0] = episode_count
        episode_rewards_log[episode_count, 1] = episode_reward
        episode_rewards_log[episode_count, 2] = running_reward

        plt.figure(1)
        plt.clf()

        plt.plot(episode_rewards_log[0:episode_count, 0],
                 episode_rewards_log[0:episode_count, 1], 'b')

        plt.plot(episode_rewards_log[0:episode_count, 0],
                 episode_rewards_log[0:episode_count, 2], 'r')

        plt.draw()
        plt.pause(0.0001)
    episode_count += 1
    ########################################seventh block   


    Tot_running_Somma=np.mean(TENepisode_PercentageSomma_history_mean)
    Tot_running_Somma3=np.mean(TENepisode_PercentageSomma_history_mean3)
    Tot_min_Somma_history=np.min(TENepisode_PercentageSomma_history_min)
    Tot_max_Somma_history=np.max(TENepisode_PercentageSomma_history_max)
    Tot_running_SommaSTD=np.mean(TENepisode_PercentageSomma_history_Var)
    Tot_running_SommaSTD3=np.mean(TENepisode_PercentageSomma_history_Var3)
    
    Tot_episode_rewardMean = np.mean( TEN_reward_Somma) # media del reward di episodio
    Tot_episode_rewardSTD = np.std( TEN_reward_Somma) # Standard deviation del reward di episodio
    Tot_episode_rewardMIN=np.min( TEN_reward_Somma)#min reward di episodio
    Tot_episode_rewardMax=np.max( TEN_reward_Somma)#max reward di episodio

    Tot_rew_Mean=np.mean(TENrunning_rewardMean)
    Tot_rew_STD=np.mean(TENrunning_rewardSTD)
    Tot_rew_min=np.min(TENrunning_rewardMIN)
    Tot_rew_max=np.max(TENrunning_rewardMAX)
    ############################################################# end
    print("TENepisode_PercentageSomma_history_max",TENepisode_PercentageSomma_history_max)
    print("N_robots",N_robots,"N_Walkman",N_Walkman," reward: ", episode_reward,"running_reward",running_reward, " elapsed: ","PercentagePulizia",running_Somma,"min_Somma_history",min_Somma_history,"max_Somma_history",max_Somma_history)
    
    ##############################################################################################8th block
    with open("ReportDatiMerakiColonyNew"+str(N_robots)+str(N_Walkman)+".txt", "a") as f:
        print("Fine 50 Episodi: N_robots",N_robots,"N_Walkman",N_Walkman,"Tot_episode_rewardMax",Tot_episode_rewardMax,"Tot_episode_rewardMIN",Tot_episode_rewardMIN,"Tot_episode_rewardSTD",Tot_episode_rewardSTD,"Tot_episode_rewardMean",Tot_episode_rewardMean,"episode_reward: ", episode_reward,"running_reward",running_reward,"Tot_rew_max",Tot_rew_max,"Tot_rew_min",Tot_rew_min,"Tot_rew_STD",Tot_rew_STD,"Tot_rew_Mean",Tot_rew_Mean,"Tot_running_Somma",Tot_running_Somma,"Tot_running_SommaSTD",Tot_running_SommaSTD,"Tot_min_Somma_history",Tot_min_Somma_history,"Tot_max_Somma_history",Tot_max_Somma_history,"Tot_running_Somma3",Tot_running_Somma3,"Tot_running_SommaSTD3",Tot_running_SommaSTD3,file=f)   
        #print("FINE 50 Episodi ",Tot_rew_max," ",Tot_rew_min," ",Tot_rew_STD," ",Tot_rew_Mean," ",Tot_running_Somma," ",Tot_running_SommaSTD," ",Tot_min_Somma_history," ",Tot_max_Somma_history, file=f)
        
    ##########################################################################################################
    episode_PercentageSomma_history.clear()
    episode_PercentageSomma_history3.clear()
    ############################################### new second block
    #testato il secondo blocco, tutto ok
    TENepisode_PercentageSomma_history_max.clear()  # only to evaluate test;
    TENepisode_PercentageSomma_history_min.clear()  # only to evaluate test;
    TENepisode_PercentageSomma_history_mean.clear()  # only to evaluate test;
    TENepisode_PercentageSomma_history_Var.clear()
    TENepisode_PercentageSomma_history_mean3.clear()  # only to evaluate test;
    TENepisode_PercentageSomma_history_Var3.clear()
    StepReward_history.clear()
    TENrunning_rewardMIN.clear()
    TENrunning_rewardSTD.clear()
    TENrunning_rewardMean.clear()
    TENrunning_rewardMAX.clear()
    TEN_reward_Somma.clear()
    ##########################################################
  


if __name__ == "__main__":

    main()