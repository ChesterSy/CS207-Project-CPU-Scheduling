import numpy as np
import pandas as pd
import operator
import sys
from tabulate import tabulate

def fcfs(processes, arrival_time, burst_time):
	process = list(np.arange(1, processes+1))
	df = pd.DataFrame(list(zip(process, arrival_time, burst_time)), columns=['Process', 'Arrival Time', 'Burst Time'])

	df.sort_values(by='Arrival Time', inplace=True) #sort it by arrival time

	completion_time = dict()
	service_time = dict()
	waiting_time = dict()
	turn_around_time = dict()
	waiting_count = df.iloc[0][1]
	burstTime = 0
	current_process = df.iloc[0][0] #get the prcoess of first row in df
	arrived = False

	while True:
		index = df[df['Process'] == current_process].index.item() #get the index number in the df of the current process
		burstTime = df.loc[index][2] #get the burst time of the current process
		waiting_count += df.loc[index][2] #add the burst time to the waiting_count for later computation

		df['Arrival Time'] = df['Arrival Time'] - burstTime #decrease the arrival time of all processes by the burst time

		df.loc[index][2] = 0 #since the process is complete, burst time = 0. I think this code is uneccessary
		completion_time[current_process] = waiting_count #since its complete, record the process -> completion time
		df.drop(index, inplace=True) #since that current process is done, drop if from df

		if df.empty: #if df is already empty, means everything is already done, break out the while loop
			break

		if df.iloc[0][1] <= 0: #if the next process is <= 0, meaning it already arrived
			current_process = df.iloc[0][0] #assign the process to the current_process variable to get executed in the next iteration
		else:
			while arrived == False: #meaning no process had yet arrived, so we need to wait for some ms until a process arrived
				waiting_count += 1 #elapsed 1ms, then check if a process had arrived by that time
				df['Arrival Time'] = df['Arrival Time'] - 1

				if df.iloc[0][1] <= 0: #checking if the process had already arrived
					current_process = df.iloc[0][0] #assign the process to the current_process variable to get executed in the next iteration
					arrived = True
			arrived = False

	for proc, arr in zip(process, arrival_time):
		turn_around_time[proc] = completion_time[proc] - arr #Turnaround time = Completion time - Arrival time

	for i, burst in enumerate(burst_time):
		waiting_time[i+1] = turn_around_time[i+1] - burst #Waiting time = Turnaround time - Burst time	

	avg_waiting_time = np.mean(list(waiting_time.values()))
	avg_turn_around_time = np.mean(list(turn_around_time.values()))

	return waiting_time, turn_around_time, avg_waiting_time, avg_turn_around_time

def sjf(processes, arrival_time, burst_time):
	process = list(np.arange(1, processes+1))
	df = pd.DataFrame(list(zip(process, arrival_time, burst_time)), columns=['Process', 'Arrival Time', 'Burst Time'])

	completion_time = dict()
	waiting_time = dict()
	turn_around_time = dict()
	burst_cmp = dict()
	arrived = False
	burstTime = 0
	waiting_count = -1
	current_process = -1

	if len(set(arrival_time)) == 1: #basically if all arrival time is 0, then just sort them by burst time
		df.sort_values(by='Burst Time', inplace=True) #sort it by burst time
		waiting_count = df.iloc[0][1]
		current_process = df.iloc[0][0] #get the prcoess of first row in df
	else:
		df.sort_values(by='Arrival Time', inplace=True) #sort it by arrival time

		temp = df.iloc[0][1] #check if there are more than 1 that arrived, if yes then we get the minimum burst out of those
		for i, row in df.iterrows(): #iterate whole df
			if row['Arrival Time'] == temp: #gets all process that has equal arrival time, meaning it arrived at the same time
				burst_cmp[row['Process']] = row['Burst Time'] #place them in a dictionary for later use
		
		current_process = min(burst_cmp.items(), key=operator.itemgetter(1))[0] #from the dictionary, get the minimum burst time
		burst_cmp = dict()
		idx = df[df['Process'] == current_process].index.item()
		waiting_count = df.loc[idx][1]

	while True:
		index = df[df['Process'] == current_process].index.item() #get the index number in the df of the current process
		burstTime = df.loc[index][2] #get the burst time of the current process
		waiting_count += df.loc[index][2] #add the burst time to the waiting_count for later computation

		df['Arrival Time'] = df['Arrival Time'] - burstTime #decrease the arrival time of all processes by the burst time

		df.loc[index][2] = 0 #since the process is complete, burst time = 0. I think this code is uneccessary
		completion_time[current_process] = waiting_count #since its complete, record the process -> completion time
		df.drop(index, inplace=True) #since that current process is done, drop if from df

		if df.empty: #if df is already empty, means everything is already done, break out the while loop
			break

		df.sort_values(by='Burst Time', inplace=True) #sort it by burst time

		for i, row in df.iterrows(): #iterate whole df, to check if there's a newly arrived process
			if row['Arrival Time'] <= 0: #all arrival time <= 0, means that process had already arrived
				burst_cmp[row['Process']] = row['Burst Time'] #place them in a dictionary for later use

		if not bool(burst_cmp):
			while arrived == False: #meaning no process had yet arrived, so we need to wait for some ms until a process arrived
				waiting_count += 1 #elapsed 1ms, then check if a process had arrived by that time
				df['Arrival Time'] = df['Arrival Time'] - 1

				if df.iloc[0][1] <= 0: #checking if the process had already arrived
					current_process = df.iloc[0][0] #assign the process to the current_process variable to get executed in the next iteration
					arrived = True
			arrived = False	
			
		current_process = min(burst_cmp.items(), key=operator.itemgetter(1))[0] #from the dictionary, get the minimum burst time
		burst_cmp = dict()

	for proc, arr in zip(process, arrival_time):
		turn_around_time[proc] = completion_time[proc] - arr #Turnaround time = Completion time - Arrival time

	for i, burst in enumerate(burst_time):
		waiting_time[i+1] = turn_around_time[i+1] - burst #Waiting time = Turnaround time - Burst time

	avg_waiting_time = np.mean(list(waiting_time.values()))
	avg_turn_around_time = np.mean(list(turn_around_time.values()))

	return waiting_time, turn_around_time, avg_waiting_time, avg_turn_around_time


def prio(processes, arrival_time, burst_time, priority_num):
	process = list(np.arange(1, processes+1))
	df = pd.DataFrame(list(zip(process, arrival_time, burst_time, priority_num)), columns=['Process', 'Arrival Time', 'Burst Time', 'Priority'])

	completion_time = dict()
	waiting_time = dict()
	turn_around_time = dict()
	priority_cmp = dict()
	arrived = False
	burstTime = 0
	waiting_count = -1
	current_process = -1

	if len(set(arrival_time)) == 1:
		df.sort_values(by='Priority', ascending=True, inplace=True) #sort it by priority. asc=True == 1 highest, asc=False == 1 lowest
		waiting_count = df.iloc[0][1]
		current_process = df.iloc[0][0]
	else:
		df.sort_values(by='Arrival Time', inplace=True) #sort it by arrival time

		temp = df.iloc[0][1]	#check if there are more than 1 that arrived, if yes then we should get the minimum priority out of those
		for i, row in df.iterrows(): #iterate whole df
			if row['Arrival Time'] == temp: #gets all process that has equal arrival time, meaning it arrived at the same time
				priority_cmp[row['Process']] = row['Priority'] #place them in a dictionary for later use
		
		current_process = min(priority_cmp.items(), key=operator.itemgetter(1))[0] #"min" == 1 highest, "max" == 1 lowest
		priority_cmp = dict()
		idx = df[df['Process'] == current_process].index.item()
		waiting_count = df.loc[idx][1]

	while True:
		index = df[df['Process'] == current_process].index.item() #get the index number in the df of the current process
		burstTime = df.loc[index][2] #get the burst time of the current process
		waiting_count += df.loc[index][2] #add the burst time to the waiting_count for later computation

		df['Arrival Time'] = df['Arrival Time'] - burstTime #decrease the arrival time of all processes by the burst time

		df.loc[index][2] = 0 #since the process is complete, burst time = 0. I think this code is uneccessary
		completion_time[current_process] = waiting_count #since its complete, record the process -> completion time
		df.drop(index, inplace=True) #since that current process is done, drop if from df

		if df.empty: #if df is already empty, means everything is already done, break out the while loop
			break

		df.sort_values(by='Priority', ascending=True, inplace=True) #sort it by priority. asc=True == 1 highest, asc=False == 1 lowest

		for i, row in df.iterrows(): #iterate whole df, to check if there's a newly arrived process
			if row['Arrival Time'] <= 0: #all arrival time <= 0, means that process had already arrived
				priority_cmp[row['Process']] = row['Priority'] #place them in a dictionary for later use

		if not bool(priority_cmp):
			while arrived == False: #meaning no process had yet arrived, so we need to wait for some ms until a process arrived
				waiting_count += 1 #elapsed 1ms, then check if a process had arrived by that time
				df['Arrival Time'] = df['Arrival Time'] - 1

				if df.iloc[0][1] <= 0: #checking if the process had already arrived
					current_process = df.iloc[0][0] #assign the process to the current_process variable to get executed in the next iteration
					arrived = True
			arrived = False	
			
		current_process = min(priority_cmp.items(), key=operator.itemgetter(1))[0] #"min" == 1 highest, "max" == 1 lowest
		priority_cmp = dict()

	for proc, arr in zip(process, arrival_time):
		turn_around_time[proc] = completion_time[proc] - arr #Turnaround time = Completion time - Arrival time

	for i, burst in enumerate(burst_time):
		waiting_time[i+1] = turn_around_time[i+1] - burst #Waiting time = Turnaround time - Burst time

	avg_waiting_time = np.mean(list(waiting_time.values()))
	avg_turn_around_time = np.mean(list(turn_around_time.values()))

	return waiting_time, turn_around_time, avg_waiting_time, avg_turn_around_time


def generate_table(num_processes, waiting_time, turn_around_time, average_wait, average_turnaround):
	headers = ['Waiting time:', 'Turnaround time:']
	table = []

	for i in range(1, num_processes+1):
		row = ['Process {}: {}'.format(i,waiting_time.get(i)), 'Process {}: {}'.format(i,turn_around_time.get(i))]
		table.append(row)

	table.append(['Average Waiting Time: {}'.format(average_wait), 'Average Turnaround Time: {}'.format(average_turnaround)])	
	return table, headers

# PROGRAM START
again = 'yes'
while again in ['yes', 'y']:
	while True:
		try:
			num_processes = int(input('Input no. of processes [2-9]: '))
		except:
			continue		
		else:
			if num_processes not in range(2,10):
				continue
			else:
				break

	print('Input individual arrival time:')
	arrival_time = list()
	for i in range(num_processes):
		while True:
			try:
				arrival_time.append(int(input('Arrival Time {}: '.format(i+1))))
			except:
				continue		
			else:
				break

	print('Input individual burst time:')
	burst_time = list()
	for i in range(num_processes):
		while True:
			try:
				burst_time.append(int(input('Burst Time {}: '.format(i+1))))
			except:
				continue		
			else:
				break

	print('CPU Scheduling Algorithm:')
	print('[A] First Come First Serve (FCFS)')
	print('[B] Shortest Job First (SJF)')
	print('[C] Priority (Prio)')
	print('[D] Exit')

	cpu_sched_algo = input('Enter choice: ').lower()

	while cpu_sched_algo not in ['a','b','c','d']:
		cpu_sched_algo = input('Enter choice: ').lower()

	if cpu_sched_algo == 'a':
		wait_time, turn_time, avg_wait, avg_turn = fcfs(num_processes, arrival_time, burst_time)
		table, headers = generate_table(num_processes, wait_time, turn_time, avg_wait, avg_turn)
		print(tabulate(table, headers, tablefmt="pretty"))

		again = input('Input again (y/n)? ').lower()
		while again not in ['y', 'n', 'yes', 'no']:
			again = input('Input again (y/n)? ').lower()
	elif cpu_sched_algo == 'b':
		wait_time, turn_time, avg_wait, avg_turn = sjf(num_processes, arrival_time, burst_time)
		table, headers = generate_table(num_processes, wait_time, turn_time, avg_wait, avg_turn)
		print(tabulate(table, headers, tablefmt="pretty"))

		again = input('Input again (y/n)? ').lower()
		while again not in ['y', 'n', 'yes', 'no']:
			again = input('Input again (y/n)? ').lower()
	elif cpu_sched_algo == 'c':
		print('Input individual priority number:')
		priority_num = list()
		for i in range(num_processes):
			while True:
				try:
					priority_num.append(int(input('Priority {}: '.format(i+1))))
				except:
					continue		
				else:
					break
		wait_time, turn_time, avg_wait, avg_turn = prio(num_processes, arrival_time, burst_time, priority_num)
		table, headers = generate_table(num_processes, wait_time, turn_time, avg_wait, avg_turn)
		print(tabulate(table, headers, tablefmt="pretty"))

		again = input('Input again (y/n)? ').lower()
		while again not in ['y', 'n', 'yes', 'no']:
			again = input('Input again (y/n)? ').lower()
	elif cpu_sched_algo == 'd':
		sys.exit()