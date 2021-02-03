import numpy as np
import pandas as pd
import operator
import sys
from tabulate import tabulate

def SRTFfindWaitingTime (num_processes, arrival_time, burst_time, waiting_time):
    rt = [0] * num_processes
    for i in range (num_processes):
        rt[i] = burst_time[i]
    complete = 0
    t = 0
    minimum = 999999999
    a = 0
    check = False

    # Iterate until all processes complete
    while (complete != num_processes): 
        # Find process with minimum remaining time among the processes that  arrives till the current time` 
        for j in range(num_processes): 
            if ((arrival_time[j] <= t) and 
                (rt[j] < minimum) and rt[j] > 0): 
                minimum = rt[j] 
                a = j 
                check = True
        if (check == False): 
            t += 1
            continue
        # Reduce remaining time by one  
        rt[a] -= 1
        # Update minimum value
        minimum = rt[a]  
        if (minimum == 0):  
            minimum = 999999999
        # If a process gets completely executed  
        if (rt[a] == 0):  
            # Increment complete  
            complete += 1
            check = False
            # Find completion time of current process  
            compt = t + 1
            # Calculate waiting time  
            waiting_time[a] = (compt - burst_time[a] - arrival_time[a]) 
            if (waiting_time[a] < 0): 
                waiting_time[a] = 0     
        # Increment time  
        t += 1

def SRTFfindTurnAroundTime(num_processes, burst_time, waiting_time, turn_around_time):  
    for i in range(num_processes): 
        turn_around_time[i] = burst_time[i] + waiting_time[i]  

def SRTF(num_processes, arrival_time, burst_time):
    waiting_time = [0] * num_processes
    turn_around_time = [0] * num_processes
    total_wt = 0
    total_tat = 0

    # call functions to find waiting time and turn around time
    SRTFfindWaitingTime(num_processes, arrival_time, burst_time, waiting_time) 
    SRTFfindTurnAroundTime(num_processes, burst_time, waiting_time, turn_around_time)  
    # find total and average waiting and turn around times
    for i in range(num_processes): 
        total_wt = total_wt + waiting_time[i]  
        total_tat = total_tat + turn_around_time[i] 

    avg_wt = total_wt / num_processes
    avg_tat = total_tat / num_processes

    return waiting_time, turn_around_time, avg_wt, avg_tat

def rr(processes, arrival_time, burst_time, quantum_time):
	process = list(np.arange(1, processes+1))
	df = pd.DataFrame(list(zip(process, arrival_time, burst_time)), columns=['Process', 'Arrival Time', 'Burst Time'])
	df2 = pd.DataFrame(list(zip(process, arrival_time, burst_time)), columns=['Process', 'Arrival Time', 'Burst Time'])

	ready_queue = list()
	completion_time = dict()
	waiting_time = dict()
	turn_around_time = dict()
	gantt = 0

	for i, row in df.iterrows():
		if row['Arrival Time'] == 0:
			ready_queue.append(row['Process']) #bale kng 0 arrival time (nakarating na), then pasok na agad sa ready queue

	for process_num in ready_queue:
		i = df2[df2['Process'] == process_num].index.item()
		df2.drop(i, inplace=True) #then lahat ng process na nasa ready queue, tatanggalin na natin sa df2, kasi hindi na ntin sila uli babalikan

	while True:
		ready_queue_2 = list()
		try:
			ready = ready_queue.pop(0)
		except:
			break

		index = df[df['Process'] == ready].index.item() #kunin lng index number ng specific process
		process_burst = df.iloc[index][2] #kunin lng yng burst time ng process na yon
		if process_burst <= quantum_time: #if mas lower/equal si burst kay quantum, then burst gagamitin pangplus ksi yung lng yng time na nagamit
			gantt += process_burst #para matake note lang gano na katagal nagrurun yng ms
			df.loc[index][2] = 0 #syempre burst <= quantum, so malamang tapos na yng process na yon, kaya 0
			completion_time[ready] = gantt #since tapos na, lalagay na sa completion_time dict
		else:
			gantt += quantum_time #else if mas mataas quantum, then quantum gagamitin pangplus ksi yung yng time na nagamit
			df.loc[index][2] = process_burst - quantum_time #then burst - quantume kasi tapos na yng quantum time na yun for the burst

		for i, row in df2.iterrows(): #iterate buong table, tapos minus ko yung arrival time para alm ko if dumating na or hindi pa
			if process_burst <= quantum_time: #syempre if less than or equal si burst, yun lng ginamit na time
				row['Arrival Time'] = row['Arrival Time'] - process_burst
			else:
				row['Arrival Time'] = row['Arrival Time'] - quantum_time

		for j, row in df2.iterrows(): #lahat ng arrival na 0 or less na (meaning dumating na), append na sa ready queue
			if row['Arrival Time'] <= 0:
				ready_queue_2.append(row['Process'])

		df2.drop(df2[df2['Arrival Time'] <= 0].index, inplace=True) #remove lahat ng arrival time na <= 0

		if ready not in completion_time: #if nasa completion time dict na yung process (meaning tapos na), edi di na kelangan iappend pa uli
			ready_queue_2.append(ready)

		ready_queue.extend(ready_queue_2)

	for proc, arr in zip(process, arrival_time):
		turn_around_time[proc] = completion_time[proc] - arr #Turnaround time = Completion Time - Arrival Time

	for i, burst in enumerate(burst_time):
		waiting_time[i+1] = turn_around_time[i+1] - burst #Waiting time = Turnaround time - Burst time	

	avg_waiting_time = np.mean(list(waiting_time.values()))
	avg_turn_around_time = np.mean(list(turn_around_time.values()))

	return waiting_time, turn_around_time, avg_waiting_time, avg_turn_around_time

def pprio(processes, arrival_time, burst_time, priority_num):
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
		waiting_count += 1 #1ms had already passed
		
		df['Arrival Time'] = df['Arrival Time'] - 1 #decrease the arrival time of all processes since 1ms already elapsed

		index = df[df['Process'] == current_process].index.item() #get the index number in the df of the current process
		df.loc[index][2] -= 1 #decrease the burst time of the current process by 1, since 1ms na lumipas

		if df.loc[index][2] == 0: #if the process is already finish. If burst time == 0
			completion_time[current_process] = waiting_count #since its complete, record the process -> completion time
			df.drop(index, inplace=True) #since the current process is done, drop it from df

		for i, row in df.iterrows(): #iterate whole df, to check if may newly arrived process
			if row['Arrival Time'] <= 0: #all arrival time <= 0, means it already arrived
				priority_cmp[row['Process']] = row['Priority'] #store it in a dictionary

		if not bool(priority_cmp): #if priority_cmp is empty
			break

		current_process = min(priority_cmp.items(), key=operator.itemgetter(1))[0] #"min" == 1 highest, "max" == 1 lowest
		priority_cmp = dict() #re-initialize the priority_cmp dictionary 

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
	print('[A] Shortest Remaining Time First (SRTF)')
	print('[B] Round Robin (RR)')
	print('[C] Preemptive Priority (P-Prio)')
	print('[D] Exit')

	cpu_sched_algo = input('Enter choice: ').lower()

	while cpu_sched_algo not in ['a','b','c','d']:
		cpu_sched_algo = input('Enter choice: ').lower()

	if cpu_sched_algo == 'a':
		wait_time, turn_time, avg_wait, avg_turn = SRTF(num_processes, arrival_time, burst_time)

		wait_time_dict = dict()
		turn_time_dict = dict()
		for i, (wait, turn) in enumerate(zip(wait_time, turn_time)):
			wait_time_dict[i+1] = wait
			turn_time_dict[i+1] = turn
		table, headers = generate_table(num_processes, wait_time_dict, turn_time_dict, avg_wait, avg_turn)
		print(tabulate(table, headers, tablefmt="pretty"))

		again = input('Input again (y/n)? ').lower()
		while again not in ['y', 'n', 'yes', 'no']:
			again = input('Input again (y/n)? ').lower()

	elif cpu_sched_algo == 'b':
		while True:
			try:
				quantum = int(input('Input time slice: '))
			except:
				continue	
			else:
				break
		wait_time, turn_time, avg_wait, avg_turn = rr(num_processes, arrival_time, burst_time, quantum)
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
		wait_time, turn_time, avg_wait, avg_turn = pprio(num_processes, arrival_time, burst_time, priority_num)
		table, headers = generate_table(num_processes, wait_time, turn_time, avg_wait, avg_turn)
		print(tabulate(table, headers, tablefmt="pretty"))

		again = input('Input again (y/n)? ').lower()
		while again not in ['y', 'n', 'yes', 'no']:
			again = input('Input again (y/n)? ').lower()
	elif cpu_sched_algo == 'd':
		sys.exit()