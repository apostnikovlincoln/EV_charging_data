import cvxpy as cvx
import numpy as np
import pandas as pd
import datetime as DT
import matplotlib.pyplot as plt
import math
import pickle
from itertools import islice

np.random.seed(1)

# load all data
ev_dataset = pd.read_csv('CrowdCharge.csv', sep=',', header=0, low_memory=False, infer_datetime_format=True)
ev_capacity = ev_dataset[['CarKW','CarKWh']]
ev_connection_time = ev_dataset[['AdjustedStartTime','AdjustedStopTime']]
ev_charge_time = ev_dataset[['ActiveCharging_Start','EndCharge']]
ev_trial = ev_dataset['Trial']
ev_dataset_uncoord = ev_dataset.loc[ev_dataset['Trial'].isnull()]
ev_charging_data = ev_dataset[['ChargerID','ParticipantID','CarKW','CarKWh','AdjustedStartTime','AdjustedStopTime','ActiveCharging_Start','EndCharge','Trial']]
ev_charging_data['Trial'] = ev_charging_data['Trial'].fillna(0)
ev_charging_data = ev_charging_data.dropna()

# list unique participants with missing data
participants_ = ev_dataset['ParticipantID'].unique()
print('unique participants {}'.format(participants_.size))

# list unique participants
participants = ev_charging_data['ParticipantID'].unique()
print('unique participants with full datasets {}'.format(participants.size))

# list unique charging stations
chargers = ev_charging_data['ChargerID'].unique()

# checking how many unique chargers have been used per customer
for p in participants:
    print("unique ChargerIDs {} per participant {}".format(
              ev_charging_data.loc[ev_charging_data['ParticipantID'] == p]['ChargerID'].unique(),p))



ev_df = pd.DataFrame()
ev_df['ParticipantID'] = participants
ev_df['ChargerID'] = chargers
ev_df['PowerProfile'] = ""

# obtain charging patterns for each participant
ii = 0
for p in participants:
    total_days = 0
    total_connection = np.zeros(0)
    total_power = np.zeros(0)
    total_soc = np.zeros(0)
    total_upflex = np.zeros(0)
    total_downflex = np.zeros(0)
    trials = np.zeros(0)
    n_rows = len(ev_charging_data.loc[ev_charging_data['ParticipantID'] == p])
    
    # use None for islice to iterate without a limit and (..,start,end) to set limits
    # here, islice(..,None) is used to pickle all data, and islice(..,0,1) to pickle single day data    
    for ind, row in islice(ev_charging_data.loc[ev_charging_data['ParticipantID'] == p].iterrows(),0,1):

        charge_t1 = DT.datetime.strptime(row['ActiveCharging_Start'], '%d/%m/%Y %H:%M')
        charge_t2 = DT.datetime.strptime(row['EndCharge'], '%d/%m/%Y %H:%M')
        
        connect_t1 = DT.datetime.strptime(row['AdjustedStartTime'], '%d/%m/%Y %H:%M')
        connect_t2 = DT.datetime.strptime(row['AdjustedStopTime'], '%d/%m/%Y %H:%M')
        
        day_start = DT.datetime(connect_t1.year,connect_t1.month,connect_t1.day)
        
        connection_start = int((connect_t1-day_start).total_seconds() / 60.0)
        connection_end = int((connect_t2-day_start).total_seconds() / 60.0)
        charging_start = int((charge_t1-day_start).total_seconds() / 60.0)
        charging_end = int((charge_t2-day_start).total_seconds() / 60.0)

        activity_days = math.ceil(connection_end/1440)
        
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
        for t in range(charging_start,charging_end):
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
        
        total_power = np.concatenate((total_power,power))
    
    ev_df.at[ii,'PowerProfile'] = total_power
    ii = ii + 1
    
# pickling power data
ev_df.to_pickle('en_ev_kw_series.pkl')        

# plots aggregated data for one day
n_days = 1
aggregated_connection = np.zeros(60*24*n_days)
aggregated_power = np.zeros(60*24*n_days)
aggregated_soc = np.zeros(60*24*n_days)
aggregated_upflex = np.zeros(2*24*n_days)
aggregated_downflex = np.zeros(2*24*n_days)

for p in participants:
    total_days = 0
    total_connection = np.zeros(0)
    total_power = np.zeros(0)
    total_soc = np.zeros(0)
    total_upflex = np.zeros(0)
    total_downflex = np.zeros(0)
    trials = np.zeros(0)
    n_rows = len(ev_charging_data.loc[ev_charging_data['ParticipantID'] == p])
    
    # use None for islice to iterate without a limit and (..,start,end) to set limits
    # here, (0,1) is used to plot the aggregated profile 
    for ind, row in islice(ev_charging_data.loc[ev_charging_data['ParticipantID'] == p].iterrows(),0,n_days):

        charge_t1 = DT.datetime.strptime(row['ActiveCharging_Start'], '%d/%m/%Y %H:%M')
        charge_t2 = DT.datetime.strptime(row['EndCharge'], '%d/%m/%Y %H:%M')
        
        connect_t1 = DT.datetime.strptime(row['AdjustedStartTime'], '%d/%m/%Y %H:%M')
        connect_t2 = DT.datetime.strptime(row['AdjustedStopTime'], '%d/%m/%Y %H:%M')
        
        day_start = DT.datetime(connect_t1.year,connect_t1.month,connect_t1.day)
        
        connection_start = int((connect_t1-day_start).total_seconds() / 60.0)
        connection_end = int((connect_t2-day_start).total_seconds() / 60.0)
        charging_start = int((charge_t1-day_start).total_seconds() / 60.0)
        charging_end = int((charge_t2-day_start).total_seconds() / 60.0)
        
        activity_days = math.ceil(connection_end/1440)
        
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
        for t in range(charging_start,charging_end):
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
        
        total_connection = np.concatenate((total_connection,connection))
        total_power = np.concatenate((total_power,power))
        trials = np.concatenate((trials,trial))
        
        soc_ = soc/100
        total_soc = np.concatenate((total_soc,soc_))
        
        total_days += activity_days
        
        upflex = pd.Series(np.zeros(activity_days*24*60))
        downflex = pd.Series(np.zeros(activity_days*24*60))
        for t in range(connection_start,connection_end+1):
            upflex[t] = row['CarKW'] - power[t]
            downflex[t] = row['CarKW'] + power[t]
            
        # averaged over 30 min intervals
        avg_upflex = pd.Series(np.zeros(activity_days*24*2))
        avg_downflex = pd.Series(np.zeros(activity_days*24*2))
        for t in range(activity_days*24*2):  
            avg_upflex[t] = sum(upflex[t*30:(t+1)*30])/30
            avg_downflex[t] = sum(downflex[t*30:(t+1)*30])/30
            
        total_upflex = np.concatenate((total_upflex,avg_upflex))
        total_downflex = np.concatenate((total_downflex,avg_downflex))

    # line plot for total power and connection
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Time, minutes')
    ax1.set_ylabel('Power, kW')
            
    line0 = ax1.plot(total_power, label='EV charging power', linestyle='solid')
    line1 = ax1.plot(trials, color="green", label='Trial', linestyle='solid')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Connection state/Battery SoC, [0,1]')
    line2 = ax2.plot(total_connection, color='orange', label='EV connection state', linestyle=':')
    line3 = ax2.plot(total_soc, color='red', label='EV battery SoC', linestyle='--')
    
    lines = line0 + line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    
    ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
            ncol=2, mode="expand", borderaxespad=0.)
    fig.tight_layout()
    plt.show()
    
    aggregated_connection[0:60*24*n_days] = aggregated_connection[0:60*24*n_days] + total_connection[0:60*24*n_days]
    aggregated_power[0:60*24*n_days] = aggregated_power[0:60*24*n_days] + total_power[0:60*24*n_days]
    aggregated_upflex[0:2*24*n_days] = aggregated_upflex[0:2*24*n_days] + total_upflex[0:2*24*n_days]
    aggregated_downflex[0:2*24*n_days] = aggregated_downflex[0:2*24*n_days] + total_downflex[0:2*24*n_days]

    
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Time, minutes')
ax1.set_ylabel('Power, kW')
        
line1 = ax1.plot(aggregated_power, label='EV charging power', linestyle='solid')

ax2 = ax1.twinx()
ax2.set_ylabel('Connection state, [0,1]')
line2 = ax2.plot(aggregated_connection, color='orange', label='EV connection state', linestyle=':')

lines = line1 + line2
labels = [l.get_label() for l in lines]

ax1.legend(lines, labels, fontsize=8, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        ncol=2, mode="expand", borderaxespad=0.)

times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]
plt.xticks(range(0, 1680, 240), times)
fig.tight_layout()
plt.savefig('aggregated_profile.pdf')
plt.show()


plt.figure(1)
plt.ylabel('Power, kW', fontsize=8)
plt.xlabel('Time, 30m intervals', fontsize=8)
        
plt.plot(aggregated_upflex, label='Aggregated G2V flexibility', linestyle='solid')
plt.plot(aggregated_downflex, label='Aggregated V2G flexibility', linestyle='solid')

plt.legend(fontsize=8, loc='upper left')
times = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]
plt.xticks(range(0, 56, 8), times)
fig.tight_layout()
plt.savefig('flexibility.pdf')
plt.show()