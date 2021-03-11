'''  
本程式的目的是為了要計算出儲能系統的排程 ，完成削峰填谷的應用 
輸入(可調整)參數 :電力負載資料(loadprofile) 儲能系統規格 (電量 功率大小 初始soc)
輸出 :24小時儲能係統排程 (變數名稱 ess_power)

原理 :混合整數線性規劃，目標函數為 最小化功率變動 

'''

#region  import
# coding=utf-8
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10 
import numpy as np

'''  顯示目前環境 創建一個 model'''
import docplex
from docplex.mp.environment import Environment
env = Environment()
# env.print_information()
from docplex.mp.model import Model
#endregion 

#region 設定負載與變數 參數
ucpm = Model("ucp") #模型選擇經濟調度優化問題 The Unit Commitment Problem (UCP)

loadprofile=[30,39,37,37,37,80,50,77,100,125,125,125,125,125,110,100,93,109,92,85,78,66,53,42]
loadprofile=[100,125,125,125,125,125,110,100,93,109,92,85,78,66,53,42,30,39,37,37,37,80,50,77]
loadprofile=[130, 123, 116, 110, 105, 100, 105, 110, 123, 136, 159, 172, 155, 161, 167, 172, 167, 162, 158, 155, 150, 145, 140, 135]

print(loadprofile)

Rise_falg = False #為了要過濾首尾 建立的旗標 
fall_falg = False


for i in range(len(loadprofile)):
    loadprofile[i] = loadprofile[i]/4
load_small=[30,39,37,37,37,39,50,77,100,125,125,125,125,125,110,100,93,109,60,20,30,10,10,15]
loadprofile= Series(loadprofile)
#endregion 
#region 儲能系統參數

NOMb = 44 #標稱電池容量,單位為kWh
SOC_Init = 0.5
SOCmin = 0.1 #電池充電狀態(最小)
SOCmax =0.9
SOC_final =0.5
ESS_Ch_max = 100 #Battery max允許以MW為單位的充電功率
ESS_Disch_max = 100 #Battery max允許以MW為單位的放電功率，放電功率是正的 
efficiency=1#充放電效率都假設一樣 
ESS_disch_cost=15#磨損成本係數 每MWh需要多少錢 

#endregion 

#region 設備規格參數
nb_periods = len(loadprofile)

demand = Series(loadprofile, index = range(0, nb_periods)) #加上索引 
# print("demand", demand)

energies = ["coal", "gas", "diesel", "wind"]
df_energy = DataFrame({"co2_cost": [30, 5, 15, 0]}, index=energies)

'''有很多部機組   不同的單位有不同的特性   variable_cost 不知道是什麼東西 
 變成表格 key 會變成row  直軸的index  每個機組的名稱 最後變成直軸是每個機組的名稱 橫軸是機組特性 (最大最小功率)
 '''
all_units = ["diesel1"]
ess_index = ["ess1"]
ucp_raw_unit_data = {
        "energy": ["diesel"],
        "initial" : [demand[0]],
        "variable_cost": [1],
        }
ucp_raw_ess_data = {
        "energy": ["ess"],
        "initial" : [0],
        "max_ch": [ESS_Ch_max],
        "max_disch": [ESS_Disch_max],
        "variable_cost": [10],
        }
df_units = DataFrame(ucp_raw_unit_data, index=all_units)
ess_unit = DataFrame(ucp_raw_ess_data, index=ess_index)

df_up = pd.merge(df_units, df_energy, left_on="energy", right_index=True)
df_up.index.names=['units'] 
#加上索引名稱 
df_units.index.names=['ess_unit'] 
ess_unit.index.names=['ess_unit'] 
#設備列表 
units = all_units
ess = ["ess1"]
# 時間長度  range from 1 to nb_periods included
periods = range(0, nb_periods) 


#endregion 
#region 定義優化變數 
# 每個機組的產量是一個連續變數 
production = ucpm.continuous_var_matrix(keys1=units, keys2=periods, name="p")
production_variation = ucpm.continuous_var_matrix(keys1=units, keys2=periods, name="production_variation")
#region 儲能係統優化變數 
    #同一個時間只能充電或是放電 
charge_var = ucpm.binary_var_matrix(keys1=ess, keys2=periods, name="charge_var")
discharge_var = ucpm.binary_var_matrix(keys1=ess, keys2=periods, name="discharge_var")
    #儲能系統功率 
ess_ch_production = ucpm.continuous_var_matrix(keys1=ess, keys2=periods, name="ess_ch_production")
ess_disch_production = ucpm.continuous_var_matrix(keys1=ess, keys2=periods, name="ess_disch_production")
    #soc
ess_soc = ucpm.continuous_var_matrix(keys1=ess, keys2=periods, name="ess_soc")
#endregion 
#endregion 
# 把整個優化變數的 屬性 列印出來
# ucpm.print_information()

#region 把所有的優化變數在整合成一個表格，增加兩個index，機組名稱 與時間 每個基礎每個時間的優化變數  
df_decision_vars = DataFrame({'production': production,'production_variation': production_variation })

df_decision_vars_ess =DataFrame({'charge_var': charge_var,'discharge_var' : discharge_var,'ess_ch_production' : ess_ch_production ,'ess_disch_production' : ess_disch_production,'ess_soc':ess_soc})
# Set index names
df_decision_vars.index.names=['units', 'periods']
df_decision_vars_ess.index.names=['ess_unit', 'periods']
#endregion 


#region  最大發電量最小發電量限制
'''把df_up裡面的最大發電量最小發電量 合併過來 ，使用的方法是 join，利用共同的index  units '''
# Create a join between 'df_decision_vars' and 'df_up' Data Frames based on common index id (ie: 'units')
# In 'df_up', one keeps only relevant columns: 'min_gen' and 'm_gen'
#把電池上下功率限制 黏貼過來 
df_join_decision_vars_ess_minmax = df_decision_vars_ess.join(ess_unit[['max_ch', 'max_disch']], how='inner')
df_join_decision_vars_ess_minmax.head()

# 功率要在最大到最小之間  疊代每個行 INDEX If True 會回傳每行的第一個 疊代每一個機組的每一個小時  
for item in df_join_decision_vars_ess_minmax.itertuples(index=False):
    ucpm += (item.ess_disch_production <= item.max_disch * item.discharge_var)
    ucpm += (item.ess_disch_production >= 0)
    ucpm += (item.ess_ch_production <= item.max_ch * item.charge_var)
    ucpm += (item.ess_ch_production >= 0)
    ucpm += (item.ess_soc >= SOCmin)
    ucpm += (item.ess_soc <= SOCmax)
    ucpm += (item.charge_var + item.discharge_var <= 1 ) #同時間只會充電或是放電 
#endregion 

#region soc變動限制，現在的電量會等於上個時刻的電量，加上功率流動 ，這個小時的soc 是這個小時的功率流動完之後的結果 
'''  先把優化變數表格依照幾組分組 ，取出各個機組的規格 ，迭代相鄰的小時功率，設定限制條件 '''
for ess_unit, r in df_decision_vars_ess.groupby(level='ess_unit'): #對於不同的幾組設定不同的限制
    ucpm.add_constraint(NOMb*SOC_Init - NOMb* r.ess_soc[0]  + (r.ess_ch_production[0] * efficiency) - (r.ess_disch_production[0]/efficiency) == 0) #初始化 
    for (p_ch_curr, p_disch_curr, soc_curr, soc_next) in zip(r.ess_ch_production[1:],r.ess_disch_production[1:],r.ess_soc, r.ess_soc[1:]): #從第二個到最後一個 
        ucpm.add_constraint(NOMb*soc_curr - NOMb*soc_next + p_ch_curr*efficiency - p_disch_curr/efficiency == 0)
        #效率只能假設一個  
ucpm.add_constraint(ess_soc['ess1',len(loadprofile)-1] == SOC_final)

# for unit, r in df_decision_vars.groupby(level='units'): #對於不同的幾組設定不同的限制
#     for ( p_curr, p_next) in zip(r.production, r.production[1:]): #從第二個到最後一個 



#endregion 


#region 電力供需平衡   功率變動量計算 
# Enforcing demand 電力供需平衡 
# use a >= here to be more robust, 
# objective will ensure efficient production
for period, r in df_decision_vars.groupby(level='periods'):
    total_demand = demand[period] #period 1 與load 1 相同
    ctname = "ct_meet_demand_%d" % period
    ucpm.add_constraint(ucpm.sum(r.production) + 
    df_decision_vars_ess.loc['ess1',period].ess_disch_production*efficiency -
    df_decision_vars_ess.loc['ess1',period].ess_ch_production/efficiency  == total_demand, ctname)
    #避免gap 太大 ，發電量排程太多 
    # ucpm.add_constraint(ucpm.sum(r.production) + 
    # df_decision_vars_ess.loc['ess1',period].ess_disch_production*efficiency -
    # df_decision_vars_ess.loc['ess1',period].ess_ch_production/efficiency  <= total_demand+20, ctname)
    # ucpm.add_constraint(ucpm.sum(r.production) >= total_demand, ctname)
    # print(r.columns)
    # 所有機組的發電再加上儲能系統功率>= 負載功率
    if  period <len(loadprofile)-1 : #1~22
        if loadprofile[period+1]>loadprofile[period] : #用電上升並且不是最後一點 
            #並不是上升的最後一個小時 變動量計算方式 就是 P(t+1)-P(t)
            ucpm.add_constraint(df_decision_vars.loc['diesel1',period].production_variation == df_decision_vars.loc['diesel1',period+1].production -df_decision_vars.loc['diesel1',period].production)
        if loadprofile[period+1]<=loadprofile[period]  : #用電量下降 
            #if up_down_index[period-1] !=1: #並不是轉折的第一點 變動量計算方式 就是 P(t)-P(t+1)
            ucpm.add_constraint(df_decision_vars.loc['diesel1',period].production_variation == df_decision_vars.loc['diesel1',period].production -df_decision_vars.loc['diesel1',period+1].production)

# print(df_decision_vars.iloc[:, 0])
# print(df_decision_vars.iloc[:, 1])
# print()  


ucpm.print_information()
#endregion


#region 成本特性  設定目標函數 求解
'''創建了一個新的表格 包含了成本特性  設定目標函數 最小化全部的成本加在一起 
'''
# Create a join between 'df_decision_vars' and 'df_up' Data Frames based on common index ids (ie: 'units')
# In 'df_up', one keeps only relevant columns: 'fixed_cost', 'variable_cost', 'start_cost' and 'co2_cost'
df_join_obj = df_decision_vars.join(
    df_up[['variable_cost']], how='inner')
ess_unit = DataFrame(ucp_raw_ess_data, index=ess_index)
ess_unit.index.names=['ess_unit'] 
df_join_obj_ess = df_decision_vars_ess.join(
    ess_unit[['variable_cost']], how='inner')



total_variable_cost = ucpm.sum(df_decision_vars.production_variation)# iloc[:, 1] ) #功率大小會影響變動成本 

total_ess_cost = ucpm.sum(df_join_obj_ess.ess_disch_production * df_join_obj_ess.variable_cost) #


# store expression kpis to retrieve them later.
ucpm.add_kpi(total_ess_cost   , "total ess cost")
ucpm.add_kpi(total_variable_cost, "Total Variable Cost")
# ucpm.add_kpi(total_nb_used, "Total #used")

# minimize sum of all costs
ucpm.minimize(total_variable_cost)


ucpm.print_information()
#ucpm.parameters.optimalitytarget = 3
assert ucpm.solve(), "!!! Solve of the model fails" #斷定解答一定存在不然就回傳字串 
ucpm.report()
#endregion

df_prods = df_decision_vars.production.apply(lambda v: max(0, v.solution_value)).unstack(level='units')
df_ess_disch_p =df_decision_vars_ess.ess_disch_production.apply(lambda v: max(0, v.solution_value)).unstack(level='ess_unit')
df_ess_ch_p =df_decision_vars_ess.ess_ch_production.apply(lambda v: max(0, v.solution_value)).unstack(level='ess_unit')
df_ess_soc =df_decision_vars_ess.ess_soc.apply(lambda v: max(0, v.solution_value)).unstack(level='ess_unit')

print('df_prods',df_prods)
#region 畫圖區域 
fig, ax = plt.subplots(figsize=(10,10))
ar=np.array([range(24)]).T
ax.set_xticks(range(0, nb_periods))
ax.set_yticks(range(-20,100,5))
# print(len(nb_periods))
# print(len(range(1, nb_periods+1)))
# print(len(df_prods))
# xx=range(nb_periods)
ess_power = df_ess_disch_p- df_ess_ch_p
# print("type(demand)",type(demand))
# print(demand)

# print(type(df_prods))
# print(type(df_prods['diesel1']))
# print(type(ess_power['ess1']))
# print("pv_power_sun : "+str(type(pv_power_sun)))
final_load =demand.sub(ess_power['ess1'], fill_value=0)

ax.plot(demand,label='originl_load')
ax.plot(final_load,label='new_load')
# ax.bar(np.arange(24),df_prods['diesel1'],0.2,color='orange',label='diesel1')

ax.bar(np.arange(24)+0.2,ess_power['ess1'],0.2,color='blue',label='ess_power')
ax.plot(df_ess_soc['ess1']*100,label='SOC')

ax.set_title('milp total_cost '+str(int (ucpm.objective_value)))

ax.legend(prop={'size': 20})
p_variation = []
p_variation.append(0) #要讓他符合24個 
for i in range(len(final_load)):
    if i>0:
        p_variation.append(final_load[i]-final_load[i-1])

output = DataFrame({'原本load': loadprofile,'ess': ess_power['ess1'],'新load':final_load,'p_variation' :p_variation})
# print("原本load",loadprofile)
# print("ess",ess_power['ess1'])
# print("新load",final_load)
print(output)
print('sum(p_variation)',sum(p_variation))
plt.show()

# # ax.plot(df_ess_disch_p,label='ess_disch_p')
# # ax.plot(-df_ess_ch_p,label='ess_ch_p')
#endregion 





'''待辦事項  try except  '''

#多餘的CODE --------------------
# Display first few rows of joined Data Frame
# df_join_obj.head()
# df_join_obj_ess.head()

# objective
#創造成本係數表格 為了要更改每個小時的變動成本 
# ess_index_for_frame = ["ess1","ess1","ess1","ess1","ess1","ess1",
#              "ess1","ess1","ess1","ess1","ess1","ess1",
#              "ess1","ess1","ess1","ess1","ess1","ess1",
#              "ess1","ess1","ess1","ess1","ess1","ess1"]
# peroid_index = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23"]
# MaxLoad = max(loadprofile)
# '''為了要壓低尖峰負載，因此要設計一個規則讓功率越高的時候成本越大，打算利用 成本係數 = 時段用電 /最大負載 ，並且成本 = 常數 *成本係數 (每個小時加總)'''

# cost_factor_series = {"cost_factor" :[ round(loadprofile[0]/MaxLoad,2),round(loadprofile[1]/MaxLoad,2),round(loadprofile[2]/MaxLoad,2),round(loadprofile[3]/MaxLoad,2),round(loadprofile[4]/MaxLoad,2),round(loadprofile[5]/MaxLoad,2)
#                                    ,round(loadprofile[6]/MaxLoad,2),round(loadprofile[7]/MaxLoad,2),round(loadprofile[8]/MaxLoad,2),round(loadprofile[9]/MaxLoad,2),round(loadprofile[10]/MaxLoad,2),round(loadprofile[11]/MaxLoad,2)
#                                    ,round(loadprofile[12]/MaxLoad,2),round(loadprofile[13]/MaxLoad,2),round(loadprofile[14]/MaxLoad,2),round(loadprofile[15]/MaxLoad,2),round(loadprofile[16]/MaxLoad,2),round(loadprofile[17]/MaxLoad,2)
#                                    ,round(loadprofile[18]/MaxLoad,2),round(loadprofile[19]/MaxLoad,2),round(loadprofile[20]/MaxLoad,2),round(loadprofile[21]/MaxLoad,2),round(loadprofile[22]/MaxLoad,2),round(loadprofile[23]/MaxLoad,2) ] }
# cost_Factor_frame = DataFrame(cost_factor_series,index=[ess_index_for_frame,peroid_index])
# cost_Factor_frame.index.names= ['ess_unit','periods'] 
# cost_Factor_frame.head()
# #把滲透率關聯到儲能系統的變動成本 
# for period, r in df_join_obj.groupby(level='periods'):
#   df_join_obj.at[('diesel1',period),'variable_cost'] =df_join_obj.at[('diesel1',period),'variable_cost']+ cost_Factor_frame.at[('ess1',str(period)) ,'cost_factor']*70
