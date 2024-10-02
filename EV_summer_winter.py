import cvxpy as cvx
import numpy as np
import pandas as pd
import datetime as DT
import matplotlib.pyplot as plt
import math
import pickle
import re
from itertools import islice

np.random.seed(1)

# load the list of EV ids on test network
ev_id_list = pd.read_csv('EV_id_list.csv', sep=',', header=0, low_memory=False, infer_datetime_format=True)
ev_ids = ev_id_list['EV_ID'].unique()

# advance = pd.read_csv('advance.csv', sep=',', header=0, low_memory=False, infer_datetime_format=True)

ev_dataset = pd.read_csv('CrowdCharge.csv', sep=',', header=0, low_memory=False, infer_datetime_format=True)
ev_capacity = ev_dataset[['CarKW','CarKWh']]
ev_connection_time = ev_dataset[['AdjustedStartTime','AdjustedStopTime']]
ev_charge_time = ev_dataset[['ActiveCharging_Start','EndCharge']]
ev_trial = ev_dataset['Trial']
ev_dataset_uncoord = ev_dataset.loc[ev_dataset['Trial'].isnull()]
ev_charging_data = ev_dataset[['ChargerID','ParticipantID','CarKW','CarKWh','AdjustedStartTime','AdjustedStopTime','Weekday_or_Weekend','ActiveCharging_Start','EndCharge','Trial']]
ev_charging_data['Trial'] = ev_charging_data['Trial'].fillna(0)
ev_charging_data = ev_charging_data.dropna()

ev_network_dataset = []
ev_summer_weekday = []
ev_summer_weekend = []
ev_winter_weekday = []
ev_winter_weekend = []

# remove irrelevant EV ids and filter data by season
for i in range(len(ev_ids)):
    ev_network_dataset.append(ev_charging_data[ev_charging_data['ParticipantID']==ev_ids[i]])
    
    ev_summer_weekday.append(ev_network_dataset[i][ev_network_dataset[i]['Weekday_or_Weekend']=='Weekday'])
    ev_summer_weekday[i] = ev_summer_weekday[i][ev_summer_weekday[i]['AdjustedStartTime'].str.contains('/06/|/07/|/08/')]
    
    ev_summer_weekend.append(ev_network_dataset[i][ev_network_dataset[i]['Weekday_or_Weekend']=='Weekend'])
    ev_summer_weekend[i] = ev_summer_weekend[i][ev_summer_weekend[i]['AdjustedStartTime'].str.contains('/06/|/07/|/08/')]
    
    ev_winter_weekday.append(ev_network_dataset[i][ev_network_dataset[i]['Weekday_or_Weekend']=='Weekday'])
    ev_winter_weekday[i] = ev_winter_weekday[i][ev_winter_weekday[i]['AdjustedStartTime'].str.contains('/12/|/01/|/02/')]
    
    ev_winter_weekend.append(ev_network_dataset[i][ev_network_dataset[i]['Weekday_or_Weekend']=='Weekend'])
    ev_winter_weekend[i] = ev_winter_weekend[i][ev_winter_weekend[i]['AdjustedStartTime'].str.contains('/12/|/01/|/02/')]

ev_swd_sp = np.zeros((len(ev_ids),2*24,3))
ev_swe_sp = np.zeros((len(ev_ids),2*24,3))
ev_wwd_sp = np.zeros((len(ev_ids),2*24,3))
ev_wwe_sp = np.zeros((len(ev_ids),2*24,3))

########## SUMMER WEEKDAY ################################################################################################
activity_days = 1

aggregated_connection = np.zeros(60*24*activity_days)
aggregated_power = np.zeros(60*24*activity_days)
aggregated_soc = np.zeros(60*24*activity_days)

# summer weekdays
for i in range(len(ev_ids)):
    avg_connection = np.zeros(60*24*activity_days)
    avg_power = np.zeros(60*24*activity_days)
    avg_soc = np.zeros(60*24*activity_days)
    fetch_flag = False
    for ind, row in islice(ev_summer_weekday[i].iterrows(),None):
        charge_t1 = DT.datetime.strptime(row['ActiveCharging_Start'], '%d/%m/%Y %H:%M')
        charge_t2 = DT.datetime.strptime(row['EndCharge'], '%d/%m/%Y %H:%M')
        
        connect_t1 = DT.datetime.strptime(row['AdjustedStartTime'], '%d/%m/%Y %H:%M')
        connect_t2 = DT.datetime.strptime(row['AdjustedStopTime'], '%d/%m/%Y %H:%M')
        
        day_start = DT.datetime(connect_t1.year,connect_t1.month,connect_t1.day)
        
        connection_start = int((connect_t1-day_start).total_seconds() / 60.0)
        connection_end = int((connect_t2-day_start).total_seconds() / 60.0)
        charging_start = int((charge_t1-day_start).total_seconds() / 60.0)
        charging_end = int((charge_t2-day_start).total_seconds() / 60.0)
        
        #activity_days = math.ceil(connection_end/1440)
        
        # create and populate EV profiles
        connection = pd.Series(np.zeros(activity_days*24*60))
        connection[connection_start:connection_end+1] = 1   
        
        power = pd.Series(np.zeros(activity_days*24*60))
        power[charging_start:charging_end+1] = row['CarKW']
        
        trial = np.ones(activity_days*24*60)*row['Trial']
        
        soc = pd.Series((np.zeros(activity_days*24*60)))
        l = 0.001 # decay constant
        flag = 0
        t0 = 0
        soc_init = (1+np.random.rand())*30
        soc[connection_start:charging_start] = soc_init
        for t in range(charging_start,min(charging_end,activity_days*1440)):
            soc_charge = (power[t]/(row['CarKWh']*60))*100
            if connection_start < charging_start:
                soc[t] = soc[t-1] + soc_charge
            else:
                if t == charging_start:
                    soc[t] = soc_init + soc_charge
                else:
                    soc[t] = soc[t-1] + soc_charge
            if soc[t] > 80:
                flag = 1
            if soc[t] > 80 and t0 == 0:
                t0 = t
            if flag == 1:
                power[t+1] = row['CarKW']*math.exp(-l*(t-t0)*60)
        soc[t:connection_end] = soc[t]
        
        soc = soc/100
        
        if(len(power)>1440):
            power = power[0:-1]
        
        if(fetch_flag == False):
            if('/07/' in row['AdjustedStartTime']):
                ev_swd_sp[i,:,0] = connection[0::30]
                ev_swd_sp[i,:,1] = power[0::30]
                ev_swd_sp[i,:,2] = soc[0::30]*row['CarKWh']
                fetch_flag = True
            elif('/08/' in row['AdjustedStartTime']):
                ev_swd_sp[i,:,0] = connection[0::30]
                ev_swd_sp[i,:,1] = power[0::30]
                ev_swd_sp[i,:,2] = soc[0::30]*row['CarKWh']
                fetch_flag = True
            elif('/06/' in row['AdjustedStartTime']):
                ev_swd_sp[i,:,0] = connection[0::30]
                ev_swd_sp[i,:,1] = power[0::30]
                ev_swd_sp[i,:,2] = soc[0::30]*row['CarKWh']
                fetch_flag = True
                
        # # line plot for total power and connection
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.set_xlabel('Time, minutes')
        # ax1.set_ylabel('Power, kW')
            
        # line0 = ax1.plot(power, label='EV charging power', linestyle='solid')
        # line1 = ax1.plot(trial, color="green", label='Trial', linestyle='solid')
    
        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Connection state/Battery SoC, [0,1]')
        # line2 = ax2.plot(connection, color='orange', label='EV connection state', linestyle=':')
        # line3 = ax2.plot(soc, color='red', label='EV battery SoC', linestyle='--')
    
        # lines = line0 + line1 + line2 + line3
        # labels = [l.get_label() for l in lines]
    
        # ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #     ncol=2, mode="expand", borderaxespad=0.)
        # times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
        # plt.xticks(range(0, 1680*activity_days, 240), times*activity_days + ["00:00"], rotation='vertical')
        # ax1.tick_params(axis='x', rotation=90)
        # fig.tight_layout()
        # plt.show()
        
        avg_connection[0:60*24*activity_days] = avg_connection[0:60*24*activity_days] + connection[0:60*24*activity_days]
        avg_power[0:60*24*activity_days] = avg_power[0:60*24*activity_days] + power[0:60*24*activity_days]
        
    avg_connection = (avg_connection/len(ev_summer_weekday[i]))
    avg_power = avg_power/len(ev_summer_weekday[i])
    
    aggregated_connection = aggregated_connection + avg_connection
    aggregated_power =  aggregated_power + avg_power
    
    # # line plot for total power and connection
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_xlabel('Time, minutes')
    # ax1.set_ylabel('Power, kW')
        
    # line0 = ax1.plot(avg_power, label='avg. EV charging power', linestyle='solid')

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Connection')
    # line1 = ax2.plot(avg_connection, color='orange', label='avg. Connection state', linestyle=':')

    # lines = line0 + line1
    # labels = [l.get_label() for l in lines]

    # ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #     ncol=2, mode="expand", borderaxespad=0.)
    # plt.title(ev_summer_weekday[i]['ParticipantID'].iloc[0])
    
    # times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
    # plt.xticks(range(0, 1680*activity_days, 240), times*activity_days + ["00:00"], rotation='vertical')
    # ax1.tick_params(axis='x', rotation=90)
    # fig.tight_layout()
    # plt.show()

aggregated_connection = aggregated_connection/len(ev_ids)
aggregated_power = aggregated_power

# line plot for total power and connection
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Time, minutes')
ax1.set_ylabel('Power, kW')
    
line0 = ax1.plot(aggregated_power, label='power', linestyle='solid')

ax2 = ax1.twinx()
ax2.set_ylabel('Connection')
line1 = ax2.plot(aggregated_connection, color='orange', label='connection state', linestyle=':')

lines = line0 + line1
labels = [l.get_label() for l in lines]

ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=2, mode="expand", borderaxespad=0.)
plt.title('summer weekday')

times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
plt.xticks(range(0, 1680*activity_days, 240), times*activity_days + ["00:00"], rotation='vertical')
ax1.tick_params(axis='x', rotation=90)
fig.tight_layout()
plt.show()

########## SUMMER WEEKEND ################################################################################################
activity_days = 1

aggregated_connection = np.zeros(60*24*activity_days)
aggregated_power = np.zeros(60*24*activity_days)
aggregated_soc = np.zeros(60*24*activity_days)

# summer weekends
for i in range(len(ev_ids)):
    avg_connection = np.zeros(60*24*activity_days)
    avg_power = np.zeros(60*24*activity_days)
    avg_soc = np.zeros(60*24*activity_days)
    fetch_flag = False    
    for ind, row in islice(ev_summer_weekend[i].iterrows(),None):
        charge_t1 = DT.datetime.strptime(row['ActiveCharging_Start'], '%d/%m/%Y %H:%M')
        charge_t2 = DT.datetime.strptime(row['EndCharge'], '%d/%m/%Y %H:%M')
        
        connect_t1 = DT.datetime.strptime(row['AdjustedStartTime'], '%d/%m/%Y %H:%M')
        connect_t2 = DT.datetime.strptime(row['AdjustedStopTime'], '%d/%m/%Y %H:%M')
        
        day_start = DT.datetime(connect_t1.year,connect_t1.month,connect_t1.day)
        
        connection_start = int((connect_t1-day_start).total_seconds() / 60.0)
        connection_end = int((connect_t2-day_start).total_seconds() / 60.0)
        charging_start = int((charge_t1-day_start).total_seconds() / 60.0)
        charging_end = int((charge_t2-day_start).total_seconds() / 60.0)
        
        #activity_days = math.ceil(connection_end/1440)
        
        # create and populate EV profiles
        connection = pd.Series(np.zeros(activity_days*24*60))
        connection[connection_start:connection_end+1] = 1   
        
        power = pd.Series(np.zeros(activity_days*24*60))
        power[charging_start:charging_end+1] = row['CarKW']
        
        trial = np.ones(activity_days*24*60)*row['Trial']
        
        soc = pd.Series((np.zeros(activity_days*24*60)))
        l = 0.001 # decay constant
        flag = 0
        t0 = 0
        soc_init = (1+np.random.rand())*30
        soc[connection_start:charging_start] = soc_init
        for t in range(charging_start,min(charging_end,activity_days*1440)):
            soc_charge = (power[t]/(row['CarKWh']*60))*100
            if connection_start < charging_start:
                soc[t] = soc[t-1] + soc_charge
            else:
                if t == charging_start:
                    soc[t] = soc_init + soc_charge
                else:
                    soc[t] = soc[t-1] + soc_charge
            if soc[t] > 80:
                flag = 1
            if soc[t] > 80 and t0 == 0:
                t0 = t
            if flag == 1:
                power[t+1] = row['CarKW']*math.exp(-l*(t-t0)*60)
        soc[t:connection_end] = soc[t]
        
        soc = soc/100
        
        if(len(power)>1440):
            power = power[0:-1]
        
        if(fetch_flag == False):
            if('/07/' in row['AdjustedStartTime']):
                ev_swe_sp[i,:,0] = connection[0::30]
                ev_swe_sp[i,:,1] = power[0::30]
                ev_swe_sp[i,:,2] = soc[0::30]*row['CarKWh']
                fetch_flag = True
            elif('/08/' in row['AdjustedStartTime']):
                ev_swe_sp[i,:,0] = connection[0::30]
                ev_swe_sp[i,:,1] = power[0::30]
                ev_swe_sp[i,:,2] = soc[0::30]*row['CarKWh']
                fetch_flag = True
            elif('/06/' in row['AdjustedStartTime']):
                ev_swe_sp[i,:,0] = connection[0::30]
                ev_swe_sp[i,:,1] = power[0::30]
                ev_swe_sp[i,:,2] = soc[0::30]*row['CarKWh'] 
                fetch_flag = True
        
        # # line plot for total power and connection
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.set_xlabel('Time, minutes')
        # ax1.set_ylabel('Power, kW')
            
        # line0 = ax1.plot(power, label='EV charging power', linestyle='solid')
        # line1 = ax1.plot(trial, color="green", label='Trial', linestyle='solid')
    
        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Connection state/Battery SoC, [0,1]')
        # line2 = ax2.plot(connection, color='orange', label='EV connection state', linestyle=':')
        # line3 = ax2.plot(soc, color='red', label='EV battery SoC', linestyle='--')
    
        # lines = line0 + line1 + line2 + line3
        # labels = [l.get_label() for l in lines]
    
        # ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #     ncol=2, mode="expand", borderaxespad=0.)
        # times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
        # plt.xticks(range(0, 1680*activity_days, 240), times*activity_days + ["00:00"], rotation='vertical')
        # ax1.tick_params(axis='x', rotation=90)
        # fig.tight_layout()
        # plt.show()
        
        avg_connection[0:60*24*activity_days] = avg_connection[0:60*24*activity_days] + connection[0:60*24*activity_days]
        avg_power[0:60*24*activity_days] = avg_power[0:60*24*activity_days] + power[0:60*24*activity_days]
        
    avg_connection = (avg_connection/len(ev_summer_weekend[i]))
    avg_power = avg_power/len(ev_summer_weekend[i])
    
    aggregated_connection = aggregated_connection + avg_connection
    aggregated_power =  aggregated_power + avg_power
    
    # # line plot for total power and connection
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_xlabel('Time, minutes')
    # ax1.set_ylabel('Power, kW')
        
    # line0 = ax1.plot(avg_power, label='avg. EV charging power', linestyle='solid')

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Connection')
    # line1 = ax2.plot(avg_connection, color='orange', label='avg. Connection state', linestyle=':')

    # lines = line0 + line1
    # labels = [l.get_label() for l in lines]

    # ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #     ncol=2, mode="expand", borderaxespad=0.)
    # plt.title(ev_summer_weekend[i]['ParticipantID'].iloc[0])
    
    # times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
    # plt.xticks(range(0, 1680*activity_days, 240), times*activity_days + ["00:00"], rotation='vertical')
    # ax1.tick_params(axis='x', rotation=90)
    # fig.tight_layout()
    # plt.show()

aggregated_connection = aggregated_connection/len(ev_ids)
aggregated_power = aggregated_power

# line plot for total power and connection
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Time, minutes')
ax1.set_ylabel('Power, kW')
    
line0 = ax1.plot(aggregated_power, label='power', linestyle='solid')

ax2 = ax1.twinx()
ax2.set_ylabel('Connection')
line1 = ax2.plot(aggregated_connection, color='orange', label='connection state', linestyle=':')

lines = line0 + line1
labels = [l.get_label() for l in lines]

ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=2, mode="expand", borderaxespad=0.)
plt.title('summer weekend')

times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
plt.xticks(range(0, 1680*activity_days, 240), times*activity_days + ["00:00"], rotation='vertical')
ax1.tick_params(axis='x', rotation=90)
fig.tight_layout()
plt.show()

########## WINTER WEEKDAY ################################################################################################
activity_days = 1

aggregated_connection = np.zeros(60*24*activity_days)
aggregated_power = np.zeros(60*24*activity_days)
aggregated_soc = np.zeros(60*24*activity_days)

# winter weekdays
for i in range(len(ev_ids)):
    avg_connection = np.zeros(60*24*activity_days)
    avg_power = np.zeros(60*24*activity_days)
    avg_soc = np.zeros(60*24*activity_days)
    fetch_flag = False    
    for ind, row in islice(ev_winter_weekday[i].iterrows(),None):
        charge_t1 = DT.datetime.strptime(row['ActiveCharging_Start'], '%d/%m/%Y %H:%M')
        charge_t2 = DT.datetime.strptime(row['EndCharge'], '%d/%m/%Y %H:%M')
        
        connect_t1 = DT.datetime.strptime(row['AdjustedStartTime'], '%d/%m/%Y %H:%M')
        connect_t2 = DT.datetime.strptime(row['AdjustedStopTime'], '%d/%m/%Y %H:%M')
        
        day_start = DT.datetime(connect_t1.year,connect_t1.month,connect_t1.day)
        
        connection_start = int((connect_t1-day_start).total_seconds() / 60.0)
        connection_end = int((connect_t2-day_start).total_seconds() / 60.0)
        charging_start = int((charge_t1-day_start).total_seconds() / 60.0)
        charging_end = int((charge_t2-day_start).total_seconds() / 60.0)
        
        #activity_days = math.ceil(connection_end/1440)
        
        # create and populate EV profiles
        connection = pd.Series(np.zeros(activity_days*24*60))
        connection[connection_start:connection_end+1] = 1   
        
        power = pd.Series(np.zeros(activity_days*24*60))
        power[charging_start:charging_end+1] = row['CarKW']
        
        trial = np.ones(activity_days*24*60)*row['Trial']
        
        soc = pd.Series((np.zeros(activity_days*24*60)))
        l = 0.001 # decay constant
        flag = 0
        t0 = 0
        soc_init = (1+np.random.rand())*30
        soc[connection_start:charging_start] = soc_init
        for t in range(charging_start,min(charging_end,activity_days*1440)):
            soc_charge = (power[t]/(row['CarKWh']*60))*100
            if connection_start < charging_start:
                soc[t] = soc[t-1] + soc_charge
            else:
                if t == charging_start:
                    soc[t] = soc_init + soc_charge
                else:
                    soc[t] = soc[t-1] + soc_charge
            if soc[t] > 80:
                flag = 1
            if soc[t] > 80 and t0 == 0:
                t0 = t
            if flag == 1:
                power[t+1] = row['CarKW']*math.exp(-l*(t-t0)*60)
        soc[t:connection_end] = soc[t]
        
        soc = soc/100
                
        if(len(power)>1440):
            power = power[0:-1]
        
        if(fetch_flag == False):
            if('/01/' in row['AdjustedStartTime']):
                ev_wwd_sp[i,:,0] = connection[0::30]
                ev_wwd_sp[i,:,1] = power[0::30]
                ev_wwd_sp[i,:,2] = soc[0::30]*row['CarKWh']
                fetch_flag = True
            elif('/02/' in row['AdjustedStartTime']):
                ev_wwd_sp[i,:,0] = connection[0::30]
                ev_wwd_sp[i,:,1] = power[0::30]
                ev_wwd_sp[i,:,2] = soc[0::30]*row['CarKWh']
                fetch_flag = True
            elif('/12/' in row['AdjustedStartTime']):
                ev_wwd_sp[i,:,0] = connection[0::30]
                ev_wwd_sp[i,:,1] = power[0::30]
                ev_wwd_sp[i,:,2] = soc[0::30]*row['CarKWh']
                fetch_flag = True
        
        # # line plot for total power and connection
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.set_xlabel('Time, minutes')
        # ax1.set_ylabel('Power, kW')
            
        # line0 = ax1.plot(power, label='EV charging power', linestyle='solid')
        # line1 = ax1.plot(trial, color="green", label='Trial', linestyle='solid')
    
        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Connection state/Battery SoC, [0,1]')
        # line2 = ax2.plot(connection, color='orange', label='EV connection state', linestyle=':')
        # line3 = ax2.plot(soc, color='red', label='EV battery SoC', linestyle='--')
    
        # lines = line0 + line1 + line2 + line3
        # labels = [l.get_label() for l in lines]
    
        # ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #     ncol=2, mode="expand", borderaxespad=0.)
        # times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
        # plt.xticks(range(0, 1680*activity_days, 240), times*activity_days + ["00:00"], rotation='vertical')
        # ax1.tick_params(axis='x', rotation=90)
        # fig.tight_layout()
        # plt.show()
        
        avg_connection[0:60*24*activity_days] = avg_connection[0:60*24*activity_days] + connection[0:60*24*activity_days]
        avg_power[0:60*24*activity_days] = avg_power[0:60*24*activity_days] + power[0:60*24*activity_days]
        
    avg_connection = (avg_connection/len(ev_winter_weekday[i]))
    avg_power = avg_power/len(ev_winter_weekday[i])
    
    aggregated_connection = aggregated_connection + avg_connection
    aggregated_power =  aggregated_power + avg_power
    
    # # line plot for total power and connection
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_xlabel('Time, minutes')
    # ax1.set_ylabel('Power, kW')
        
    # line0 = ax1.plot(avg_power, label='avg. EV charging power', linestyle='solid')

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Connection')
    # line1 = ax2.plot(avg_connection, color='orange', label='avg. Connection state', linestyle=':')

    # lines = line0 + line1
    # labels = [l.get_label() for l in lines]

    # ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #     ncol=2, mode="expand", borderaxespad=0.)
    # plt.title(ev_winter_weekday[i]['ParticipantID'].iloc[0])
    
    # times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
    # plt.xticks(range(0, 1680*activity_days, 240), times*activity_days + ["00:00"], rotation='vertical')
    # ax1.tick_params(axis='x', rotation=90)
    # fig.tight_layout()
    # plt.show()

aggregated_connection = aggregated_connection/len(ev_ids)
aggregated_power = aggregated_power

# line plot for total power and connection
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Time, minutes')
ax1.set_ylabel('Power, kW')
    
line0 = ax1.plot(aggregated_power, label='power', linestyle='solid')

ax2 = ax1.twinx()
ax2.set_ylabel('Connection')
line1 = ax2.plot(aggregated_connection, color='orange', label='connection state', linestyle=':')

lines = line0 + line1
labels = [l.get_label() for l in lines]

ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=2, mode="expand", borderaxespad=0.)
plt.title('winter weekday')

times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
plt.xticks(range(0, 1680*activity_days, 240), times*activity_days + ["00:00"], rotation='vertical')
ax1.tick_params(axis='x', rotation=90)
fig.tight_layout()
plt.show()

########## WINTER WEEKEND ################################################################################################
activity_days = 1

aggregated_connection = np.zeros(60*24*activity_days)
aggregated_power = np.zeros(60*24*activity_days)
aggregated_soc = np.zeros(60*24*activity_days)

# winter weekends
for i in range(len(ev_ids)):
    avg_connection = np.zeros(60*24*activity_days)
    avg_power = np.zeros(60*24*activity_days)
    avg_soc = np.zeros(60*24*activity_days)
    fetch_flag = False    
    for ind, row in islice(ev_winter_weekend[i].iterrows(),None):
        charge_t1 = DT.datetime.strptime(row['ActiveCharging_Start'], '%d/%m/%Y %H:%M')
        charge_t2 = DT.datetime.strptime(row['EndCharge'], '%d/%m/%Y %H:%M')
        
        connect_t1 = DT.datetime.strptime(row['AdjustedStartTime'], '%d/%m/%Y %H:%M')
        connect_t2 = DT.datetime.strptime(row['AdjustedStopTime'], '%d/%m/%Y %H:%M')
        
        day_start = DT.datetime(connect_t1.year,connect_t1.month,connect_t1.day)
        
        connection_start = int((connect_t1-day_start).total_seconds() / 60.0)
        connection_end = int((connect_t2-day_start).total_seconds() / 60.0)
        charging_start = int((charge_t1-day_start).total_seconds() / 60.0)
        charging_end = int((charge_t2-day_start).total_seconds() / 60.0)
        
        #activity_days = math.ceil(connection_end/1440)
        
        # create and populate EV profiles
        connection = pd.Series(np.zeros(activity_days*24*60))
        connection[connection_start:connection_end+1] = 1   
        
        power = pd.Series(np.zeros(activity_days*24*60))
        power[charging_start:charging_end+1] = row['CarKW']
        
        trial = np.ones(activity_days*24*60)*row['Trial']
        
        soc = pd.Series((np.zeros(activity_days*24*60)))
        l = 0.001 # decay constant
        flag = 0
        t0 = 0
        soc_init = (1+np.random.rand())*30
        soc[connection_start:charging_start] = soc_init
        for t in range(charging_start,min(charging_end,activity_days*1440)):
            soc_charge = (power[t]/(row['CarKWh']*60))*100
            if connection_start < charging_start:
                soc[t] = soc[t-1] + soc_charge
            else:
                if t == charging_start:
                    soc[t] = soc_init + soc_charge
                else:
                    soc[t] = soc[t-1] + soc_charge
            if soc[t] > 80:
                flag = 1
            if soc[t] > 80 and t0 == 0:
                t0 = t
            if flag == 1:
                power[t+1] = row['CarKW']*math.exp(-l*(t-t0)*60)
        soc[t:connection_end] = soc[t]
        
        soc = soc/100
                
        if(len(power)>1440):
            power = power[0:-1]
        
        if(fetch_flag == False):
            if('/01/' in row['AdjustedStartTime']):
                ev_wwe_sp[i,:,0] = connection[0::30]
                ev_wwe_sp[i,:,1] = power[0::30]
                ev_wwe_sp[i,:,2] = soc[0::30]*row['CarKWh']
                fetch_flag = True
            elif('/02/' in row['AdjustedStartTime']):
                ev_wwe_sp[i,:,0] = connection[0::30]
                ev_wwe_sp[i,:,1] = power[0::30]
                ev_wwe_sp[i,:,2] = soc[0::30]*row['CarKWh']
                fetch_flag = True
            elif('/12/' in row['AdjustedStartTime']):
                ev_wwe_sp[i,:,0] = connection[0::30]
                ev_wwe_sp[i,:,1] = power[0::30]
                ev_wwe_sp[i,:,2] = soc[0::30]*row['CarKWh']
                fetch_flag = True
        
        # # line plot for total power and connection
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.set_xlabel('Time, minutes')
        # ax1.set_ylabel('Power, kW')
            
        # line0 = ax1.plot(power, label='EV charging power', linestyle='solid')
        # line1 = ax1.plot(trial, color="green", label='Trial', linestyle='solid')
    
        # ax2 = ax1.twinx()
        # ax2.set_ylabel('Connection state/Battery SoC, [0,1]')
        # line2 = ax2.plot(connection, color='orange', label='EV connection state', linestyle=':')
        # line3 = ax2.plot(soc, color='red', label='EV battery SoC', linestyle='--')
    
        # lines = line0 + line1 + line2 + line3
        # labels = [l.get_label() for l in lines]
    
        # ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #     ncol=2, mode="expand", borderaxespad=0.)
        # times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
        # plt.xticks(range(0, 1680*activity_days, 240), times*activity_days + ["00:00"], rotation='vertical')
        # ax1.tick_params(axis='x', rotation=90)
        # fig.tight_layout()
        # plt.show()
        
        avg_connection[0:60*24*activity_days] = avg_connection[0:60*24*activity_days] + connection[0:60*24*activity_days]
        avg_power[0:60*24*activity_days] = avg_power[0:60*24*activity_days] + power[0:60*24*activity_days]
        
    avg_connection = (avg_connection/len(ev_winter_weekend[i]))
    avg_power = avg_power/len(ev_winter_weekend[i])
    
    aggregated_connection = aggregated_connection + avg_connection
    aggregated_power =  aggregated_power + avg_power
    
    # # line plot for total power and connection
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_xlabel('Time, minutes')
    # ax1.set_ylabel('Power, kW')
        
    # line0 = ax1.plot(avg_power, label='avg. EV charging power', linestyle='solid')

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Connection')
    # line1 = ax2.plot(avg_connection, color='orange', label='avg. Connection state', linestyle=':')

    # lines = line0 + line1
    # labels = [l.get_label() for l in lines]

    # ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    #     ncol=2, mode="expand", borderaxespad=0.)
    # plt.title(ev_winter_weekend[i]['ParticipantID'].iloc[0])
    
    # times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
    # plt.xticks(range(0, 1680*activity_days, 240), times*activity_days + ["00:00"], rotation='vertical')
    # ax1.tick_params(axis='x', rotation=90)
    # fig.tight_layout()
    # plt.show()

aggregated_connection = aggregated_connection/len(ev_ids)
aggregated_power = aggregated_power

# line plot for total power and connection
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Time, minutes')
ax1.set_ylabel('Power, kW')
    
line0 = ax1.plot(aggregated_power, label='power', linestyle='solid')

ax2 = ax1.twinx()
ax2.set_ylabel('Connection')
line1 = ax2.plot(aggregated_connection, color='orange', label='connection state', linestyle=':')

lines = line0 + line1
labels = [l.get_label() for l in lines]

ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=2, mode="expand", borderaxespad=0.)
plt.title('winter weekend')

times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
plt.xticks(range(0, 1680*activity_days, 240), times*activity_days + ["00:00"], rotation='vertical')
ax1.tick_params(axis='x', rotation=90)
fig.tight_layout()
plt.show()

# ############################ ADVANCE ##########################################

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_xlabel('Time, minutes')
# ax1.set_ylabel('Power, kW')
    
# line0 = ax1.plot(advance, label='advance', linestyle='solid')

# lines = line0
# labels = [l.get_label() for l in lines]

# ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
#     ncol=2, mode="expand", borderaxespad=0.)
# plt.title('Advance')

# times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"]
# plt.xticks(range(0, 56*1, 8), times*1 + ["00:00"], rotation='vertical')
# ax1.tick_params(axis='x', rotation=90)
# fig.tight_layout()
# plt.show()