from    gurobipy import *
from    itertools import product
import  itertools as it
import  xlrd
import  numpy as np
import  time
import  csv
from    itertools import permutations
import  random


FinalSolution = {}
TimeLimit     = 3600 * 12

for Obj in ["Convex"]: #"Convex","Concave","Increasing",
    for Salesman in [1]:
        for Cities in [16]: #16,31,52
            for Days in [30]: #30,45,60,75,90
                for Party in ["D","R"]: #"D","R"
                    start_time = time.process_time()
                    if Party == "D":
                        file_location = "CIPP-D.xls"
                    else:
                        file_location = "CIPP-R.xls"

                    workbook = xlrd.open_workbook(file_location)
                    sheet = workbook.sheet_by_index(0)
                    BRP = [sheet.cell_value(r,c)       for c in range(sheet.ncols)  for r in range(sheet.nrows)]      
                    sheet = workbook.sheet_by_index(4)
                    # BRP =    [random.randint(90, 100) for _ in range(Cities)]
                    ID = [sheet.cell_value(r,c)        for c in range(sheet.ncols)  for r in range(sheet.nrows)]
                    AllCities  = list(range(0,Cities))
                    AllDays    = list(range(0,90))
                    AllM       = list(range(0,10))
                    M          = AllM[:Salesman]
                    I          = AllCities[:Cities+1]
                    T          = AllDays[:Days+1]
    
                    
                    MaxMeetingsDaily           = 1 
                    MaxMeetings                = 12
                    beta1, beta2, beta3, beta4 = 2, 2, 2, 0
                    S                          = list(range(0,MaxMeetings+1))
                    alpha                      = 6
                    w                          = 2
                    penalty                    = 0.04
                    c = 10
                    
                    model  = Model("FullMILP")
                    Z  = model.addVars(M,I,T,  vtype=GRB.BINARY, name="Z")
                    X  = model.addVars(M,I,T,S,  vtype=GRB.BINARY, name="X") 
                    R  = model.addVars(M,I,S,  vtype=GRB.BINARY, name="R")
                    V  = model.addVars(T,      vtype=GRB.BINARY, name="V")
                    model.update()
                    
                    #No meeting in first day
                    model.addConstrs((Z[m,i,0]   == 0) for m,i in product(M,I))
                    # No meeting in fictitious city on any day  
                    model.addConstrs((Z[m,0,t]   == 0) for m,t in product(M,T))
    
       
                    # model.setObjective(( quicksum( X[m,i,t,s] * BRP[i] * ((len(T) - 1 - t - 1)/(len(T) - 1)) * (1-0.04*(s-1)) for m,i,t,s in product(M,I,T,S) if s>=1)),GRB.MAXIMIZE)
                                   
                    model.setObjective((quicksum( ( X[m,i,t,s] * BRP[i] * ((t-((len(T) - 1)+1)/2)**2 + ((len(T)-2)/4)**2) * (1-0.04*(s-1)) ) for m,i,t,s in product(M,I,T,S) if s>=1 )),GRB.MAXIMIZE)

    
                    #####################-------  Problem Assumptions  -------####################  
                    '''Forces the party leader to include atleast rest rest days every week'''
                    u_t = {}  #  key = day t, value = required rest days
                    
                    for t in T:
                        if t <= (len(T)-1) // 4:
                            u_t[t] = beta1
                        elif t <= (len(T)-1) // 2:
                            u_t[t] = beta2
                        elif t <= (3 * (len(T)-1)) // 4:
                            u_t[t] = beta3
                        else:
                            u_t[t] = beta4
    
                    # Piecewise rest Const 2,3
                    model.addConstrs( (quicksum(V[kk] for kk in T if kk >= t and kk <= t + alpha) >= u_t[t] for t in T if t <= len(T) - alpha), name="RestWindow" )
                    # model.addConstr(quicksum(V[t] for t in T if (len(T) - 4) <= t <= (len(T)+1)) == 0, name="RestWindow1")
    
                    # model.addConstrs(( quicksum(V[kk] for kk in T if kk >= t and kk <= t+6) >= rest for t in T if t >= 1 and t <= len(T) - 20), name="Rest")
                    model.addConstrs( ( quicksum(Z[m,i,t] for i in I) <= 1 - V[t] for m,t in product(M,T)), name="Coupling_Z_V")
        
                    # Const 10
                    '''Imposes a limit on the total number of the meetings in each REGULAR city'''
                    model.addConstrs( (quicksum( Z[m,i,t] for t in T if t >= 1)  <= MaxMeetings for m,i in product(M,I)), name="MaxNumMeet_SmallCity")
                            
                    # Const 9
                    model.addConstrs( (quicksum( Z[m,i,t] for i in I) <= MaxMeetingsDaily for m,t in product(M,T) if t >= 1), name="MaxMeetDaily")
                                    
                    '''Constraint preventing two conseutive meetings in the same city'''
                    # model.addConstrs( (Z[m,i,t] + Z[m,i,t+1] <= 1  
                    #                                     for m,i,t in product(M,I,T) if t >= 1 and i >= 1 
                    #                                     and t < len(T) - 1), name="No_Two_Consecutive_Meetings")
                    # Const 11
                    '''Constraint preventing more than two meetings every 10 days'''
                    model.addConstrs((quicksum(Z[m,i,k] for k in T if k>=t and k <= t+alpha) <= w  
                                 for m,i,t in product(M,I,T) if t >= 1 and t <= len(T)-alpha), name="Max_Two_Meetings_every_10_days")
                    
    
                    # Const 4-8
                    # Constraint: Each i has exactly one s assigned
                    model.addConstrs((quicksum(R[m, i, s] for s in S) == 1 for m,i in product(M,I)), name="One_S_Per_I")
                    
                    # Constraint: Total Z_it equals s * R_is for each i
                    model.addConstrs((quicksum(Z[m, i, t] for t in T) == quicksum(s * R[m, i, s] for s in S) for m,i in product(M,I)), name="Z_Equals_S_R")
                    
                    # Constraint: X_its <= R_is
                    model.addConstrs((X[m, i, t, s] <= R[m, i, s] for m, i, t, s in product(M, I, T, S)), name="X_Leq_R")
                    
                    # Constraint: X_its <= Z_it
                    model.addConstrs((X[m, i, t, s] <= Z[m, i, t] for m ,i, t, s in product(M, I, T, S)), name="X_Leq_Z")
                    
                    # Constraint: X_its >= R_is + Z_it - 1
                    model.addConstrs((X[m, i, t, s] >= R[m, i, s] + Z[m, i, t] - 1 for m, i, t, s in product(M, I, T, S)), name="X_Geq_R_Z")
                                    
                    # Const 12
                    # model.addConstr(sum(c * Z[m,i, t] for m in M for i in I for t in T) <= 80, name="Budget")
    
    
                    #----------------------------------------------------------------------------#
        
                    model.update()
                    model.Params.Threads = 28
                    model.Params.logfile = "%s_%s_%s.log"%(file_location,Cities,Days)
                    model.params.TimeLimit = TimeLimit
                    model.optimize()
                    runtime1 = model.runtime
                    
                    ######################-------  Print the Output  -------######################            
                    if model.status == GRB.Status.INF_OR_UNBD:
                        # Turn presolve off to determine whether model is infeasible or unbounded
                        model.setParam(GRB.Param.Presolve, 0)
                     
                    if model.status == GRB.Status.OPTIMAL:
                        print('###############################')
                        print("Runtime = %s (sec) " % round(model.runtime,2) , '\n')
                        print('Optimal objective: %g' % model.objVal, '\n')
                        print('Iterations: %g' % model.IterCount)
                        print('###############################')
                           
                    if model.status == GRB.Status.INFEASIBLE:
                        print('Optimization was stopped with status %d' % model.status)
                        model.computeIIS()
                        model.write('model.ilp')
                    #----------------------------------------------------------------------------#    
                    if  model.objval > 1:
                        Vars = []
                        for v in model.getVars():
                            if v.x > 0.1 and v.x <= 1.1 and v.varName[0] != "L" and v.varName[0] != "E" and v.varName[0] != "U":        
                                Vars.append([v.varName,v.x])   
        
                        AllTour      = [[[] for i in range(Days)]for m in M ]
                        AllMeet      = [[[] for i in range(Days)]for m in M ]
                        AllID        = [[[] for i in range(Days)]for m in M ]
                        AllMeetDict = {}              
                        for m in M:
                            for i in I:
                                for t in T:
                                    if Z[m,i,t].X > 0.9:
                                        AllMeetDict[t] = i
                                        AllMeet[m][t-1].append(i)
                        Rewards = {}
                        
                        file_location = file_location
                        workbook = xlrd.open_workbook(file_location)
                        sheet = workbook.sheet_by_index(0)
                        BRP2 = [sheet.cell_value(r,c)       for c in range(sheet.ncols)  for r in range(sheet.nrows)]
                        
    
                        def flatten_list(nested_list):
                            flat_list = []
                            for item in nested_list:
                                if isinstance(item, list):
                                    flat_list.extend(flatten_list(item))
                                else:
                                    flat_list.append(item)
                            return flat_list
                        MeetFlat = flatten_list(AllMeet)
                        count_dict = {i: MeetFlat.count(i) for i in I}
    
                        MappedObj = 0
                        for idx,val in AllMeetDict.items():
                            ss = sum(1 for value in AllMeetDict.values() if value ==val)
                            MappedObj += BRP2[val]  * ((idx-((len(T)-1 )+1)/2)**2 + ((len(T)-2)/4)**2) * (1-0.04*(ss-1))
    
                        
                        for city in I:
                            for idx,day in enumerate(AllMeet[0]):
                                if city in day:
                                    if city in Rewards: # t -> idx+1
                                        ss = idx + 1 - Rewards[city][-1][0]
                                        Rewards[city].append((idx+1,round(BRP[city]*((len(T) - (idx+1))/(len(T) - 1))*(ss/(len(T)-1)),2)))
                                    else:
                                        Rewards[city] = [(idx+1,    round(BRP[city]*((len(T) - (idx+1))/(len(T) - 1)),2))]                                
        
                        RewardsID = {}
                        for i in Rewards:
                             RewardsID[ID[i]] = Rewards[i]
                            
                        AllMeet1D  = [item[0] if item else 0 for sublist in AllMeet for item in sublist]
                        RewardsDay = [0 for t in T]
                        for idx,t in enumerate(AllMeet1D):
                            RewardsDay[idx] = (idx+1,ID[t])
                        # calculate reward with original objective
                        aa = 0
                        for value_list in Rewards.values():
                            for _, num in value_list:
                                aa += num                    
    
                        file_location = file_location
                        workbook = xlrd.open_workbook(file_location)
                        sheet = workbook.sheet_by_index(0)
                        BRP2 = [sheet.cell_value(r,c)       for c in range(sheet.ncols)  for r in range(sheet.nrows)]
                        Rewards = {}
    
                        for city in I:
                            for idx,day in enumerate(AllMeet[0]):
                                if city in day:
                                    if city in Rewards: # t -> idx+1
                                        ss = idx + 1 - Rewards[city][-1][0]
                                        Rewards[city].append((idx+1,round((BRP2[city]*(ss/(len(T)-1))*((len(T) - idx-1)/(len(T) - 1))),2)))
                                    else:
                                        Rewards[city] = [(idx+1,round((BRP2[city]*((len(T) - idx-1)/(len(T) - 1))),2))]
    
                        RewardsID = {}
                        for i in Rewards:
                             RewardsID[ID[i]] = Rewards[i]
    
                        RewardMappedwGod = 0
                        for value_list in Rewards.values():
                            for _, num in value_list:
                                RewardMappedwGod += num
    
                        TotMeetNum = 0
                        for m in M:
                            for i in range(len(AllMeet[m])):
                                TotMeetNum += len(AllMeet[m][i])
    
                        import pandas as pd
                        filtered_data = [item for item in RewardsDay if isinstance(item, tuple)]
                        days = [item[0] for item in filtered_data]
                        states = [item[1] for item in filtered_data]
                        
                        # # Count occurrences of each state
                        state_counts = {state: states.count(state) for state in set(states)}
                        counts = [state_counts[state] for state in states]
                        brp_values = [BRP[ID.index(state)] for state in states]
                        print(state_counts)
    
    
                                             
                        runtime    = str(round(runtime1,2)) 
                        objective  = str(model.getObjective)
                        Objective  = round(model.objval,2)
                        gap        = str(round(model.MIPGap*100,2))+'%' 
                        UpperBound = round(model.objbound,2)  
                        s          = 'Cities=%s | Days=%s | Num_Meetings=%s | Gap=%s | CPU(s)=%s | Obj=%s | ObjwithOriginalReward=%s | UB =%s | # of Constraints=%d | # of Variables=%d; \n \n' %(Cities, Days, TotMeetNum, gap, runtime,Objective, MappedObj, UpperBound, model.numConstrs,model.numVars)
                        # with open('%s_%s_%s.csv' %(Obj,Cities,Days), 'w', newline='') as f:

                            
                        RewardsDay = [item for item in RewardsDay if isinstance(item, tuple) and len(item) == 2]



                        with open('%s_%s_%s.csv' %(Party,Cities,Days), 'w', newline='') as f:
    
                           f.write(s)
                           writer = csv.writer(f, delimiter=',')
                           for key, value in Rewards.items(): 
                               writer.writerow([key, value])
                           for key, value in Rewards.items(): 
                               writer.writerow([key, value]) 
                           for key, value in state_counts.items():
                               writer.writerow([key, value])
                           for day, state in RewardsDay:
                               writer.writerow([day, state])
                           f.close()
                        model.write('%s_%s.sol'%(Cities,Days))
                        model.write('%s_%s.lp' %(Cities,Days))        
                        FinalSolution['%s_%s_%s'  %(Party,Cities,Days)] = [round(model.objVal,2),round(model.objbound,2),round(model.MIPGap*100,2),round(runtime1,2)]
                        print(FinalSolution)
                        
                        print(RewardsDay)
                    else:
                        FinalSolution['%s_%s_%s'  %(Party,Cities,Days)] = ' no solution'
              
with open("FinalSolution.txt", 'w') as f: 
    for key, value in FinalSolution.items(): 
        f.write('%s:%s\n' % (key, value))
        
            
            


# # Pad ID and BRP lists to match the length of the main data (65 rows)
# n_rows = len(filtered_data)  # 65 rows
# id_column = ID + [np.nan] * (n_rows - len(ID))  # Pad with NaN
# brp_column = BRP + [np.nan] * (n_rows - len(BRP))  # Pad with NaN

# # Create DataFrame
# df = pd.DataFrame({
#     'Day': days,
#     'State': states,
#     'Count': counts,
#     'BRP': brp_values,
#     'ID_List': id_column,
#     'BRP_List': brp_column
# })

# # Save to Excel
# df.to_excel('states_with_counts.xlsx', index=False)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
