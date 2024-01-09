#!/D:/Python27_64/python.exe

#############################################################################################################
#The purpose of this script is to automate the creation of response curves and
# an optimized plan for Kaleidoscope
#
#INPUT: a JSON user query called Parms.json in the working_directory_path
#OUTPUT: Three files, with the formats
# (1) curves: Placement_ID|placement_name|alpha|beta|max_x_val|max_y_val|recommended_weekly_spend|calc_type|n
# (2) report.zip: optimized spend plan(s); Channel|PlacementID|PlacementName|Package|TotalSpend
# (3) log file: detailed diagnostics, inputs/outputs, errors, etc.
#
#Authors: Chris Strong strongnyc2@gmail.com
#Date: 12/12/17, last update 2/26/22 1405
#
#This script should be run from the command line, e.g.:
#>D:\Python27_64\python.exe E:\Kaleidoscope\scripts\optimizer.py E:\Kaleidoscope\data\<run number>\
#############################################################################################################

import json
import sys
import re
import time as t
import pyodbc
import numpy as np
import zipfile
import os
from scipy.optimize import curve_fit
from scipy.optimize import minimize
np.set_printoptions(formatter={'float_kind':'{:.2f}'.format})

#################################################################################
##PARAMETERS##
#working_directory_path = 'E:\\users\\jhaynes\\kaleidoscope\\data\\A1\\'
server_connection_string = 'DRIVER={SQL Server};SERVER=50.31.135.69;DATABASE=DW_BestBuy;UID=jonhayn;PWD=Starcom.1'
placement_budget_limits = True #never allow more than a 2x budget increase from the most ever spent in the past
verbose = False
##################################################################################
start = t.time()

#use inputs from command line, if provided, otherwise use below
try:
	working_directory_path = str(sys.argv[1]) #example: E:\Kaleidoscope\data\243895798579835739857\'
except:
	print 'working_directory_path=', str(working_directory_path)
	print 'ERROR: JSON path not found\n'
	#print 'verbose=', str(verbose)

class D_simple (dict):
	def __init__(self, default=None):
		self.default = default
	def __getitem__(self,key):
		if not self.has_key(key):
			self[key] = ''
		return dict.__getitem__(self,key)
		
class D_list (dict):
	def __init__(self, default=None):
		self.default = default
	def __getitem__(self,key):
		if not self.has_key(key):
			self[key] = []
		return dict.__getitem__(self,key)

def read_json(working_directory_path, verbose):
	json_path = working_directory_path + 'parms.json'
	fh = open(json_path, 'r')
	json_data = ''
	for line in fh:
		json_data+=line
	json_data = json_data.replace('\n','')
	json_data = json_data.replace('\xef\xbb\xbf','')
	json_data = json_data.replace('\xe2\x80\x9d','')
	json_data = json.loads(json_data)
	json_data = json_data[0]
	return json_data
	
def return_sql_parameters(data):
	table = D_simple (dict)
	
	#Campaign Goal =; Awareness->Reach field, Conversion->True_Online_Revenue, Consideration->Clicks; CappedCost spend for all goals
	table['MDM_CA_Campaign_Objective'] = str(data['Campaign_Goal']) #DONE MDM_CA_Campaign_Optimization {Conversion, Consideration, xxx}
	
	#Campaign Length Intervals
	campaign_length_weeks = int(data['Campaign_Length'][:2]) #need to calculate from Prisma_CampaignStartDate and Prisma_CampaignEndDate
	if campaign_length_weeks <= 4:
		campaign_length_max = '4'
		table['Campaign_Length_Range'] = "(DATEDIFF(week,a.Prisma_CampaignStartDate,a.Prisma_CampaignEndDate) <= " + campaign_length_max + ")"
	elif campaign_length_weeks >= 5 and campaign_length_weeks <=10:
		campaign_length_min = '5'
		campaign_length_max = '10'
		table['Campaign_Length_Range'] = "(DATEDIFF(week,a.Prisma_CampaignStartDate,a.Prisma_CampaignEndDate) >= " + \
	      campaign_length_min + " AND DATEDIFF(week,a.Prisma_CampaignStartDate,a.Prisma_CampaignEndDate) <= " + campaign_length_max + ")"
	elif campaign_length_weeks >= 11 and campaign_length_weeks <=15:
		campaign_length_min = '10'
		campaign_length_max = '15'
		table['Campaign_Length_Range'] = "(DATEDIFF(week,a.Prisma_CampaignStartDate,a.Prisma_CampaignEndDate) >= " + \
	      campaign_length_min + " AND DATEDIFF(week,a.Prisma_CampaignStartDate,a.Prisma_CampaignEndDate) <= " + campaign_length_max + ")"
	elif campaign_length_weeks >= 16:
		campaign_length_min = '16'
		table['Campaign_Length_Range'] = "(DATEDIFF(week,a.Prisma_CampaignStartDate,a.Prisma_CampaignEndDate) >= " + campaign_length_min + ")"
	
	#Campaign Type; multi possible IN PZN, Social, Full Funnel
	#Counts: MDM_CA_Campaign_Type,count: Back to College 6467, Back to College Standalone Social 45, Standalone Social 147, Full Funnel 33417
	#Programmatic Only 179, Entry Level Display 65, Best Buy Standalone PZN 292, Standalone PZN 11969
	if ',' not in str(data['Campaign_Type']):
		table['MDM_CA_Campaign_Types'] = str(data['Campaign_Type'])
	else:
		table['MDM_CA_Campaign_Types'] = str(data['Campaign_Type'][0])	

	#Campaign Names IN campaign1, campaign2, etc.; updated 1/29/18
	if ',' not in str(data['Campaign_Name']):
		table['CampaignNames'] = str(data['Campaign_Name']) #if a single campaign is entered
	else:
		table['CampaignNames'] = str(data['Campaign_Name'][0]) #case where is multiple campaigns in a list, not just one
	
	#Channels IN {Social,Display,Video}
	if ',' in str(data['Channels']): # it's multiple channels in a list, i.e., [u'Video,Display']
		table['MDM_PL_Channels'] = str(data['Channels'][0])
	else:
		table['MDM_PL_Channels'] = str(data['Channels'])
	
	#Vendor Names IN
	if ',' in str(data['MDF_Vendors']):
		table['MDM_CA_Vendor_Names'] = str(data['MDF_Vendors'][0]) #MDM_CA_Vendor_Name IN csv list
	else:
		table['MDM_CA_Vendor_Names'] = str(data['MDF_Vendors'])
	
	#Optimization Type {'Open Optimization', 'Breakout Optimization'}
	optimization_type = str(data['Optimization_Type'])
	table['Optimization_Type'] = optimization_type
	
	#Optimization Budget; List of csv separated numbers, e.g., '500000,200000'; only one number if it is an Open Optimiation
	# budget amounts correspond to channels list in same order
	if ',' not in str(data['Optimization_Budget']):
		table['Optimization_Budget'] = str(data['Optimization_Budget'])
	else:
		table['Optimization_Budget'] = str(data['Optimization_Budget'][0])
	
	#optimization level; updated 1/26/18; note, MMD is a typo in the SQL table header, so this is intentional below
	optimization_level= str(data['Optimization_Level'])
	if str(data['Channels']) == 'Social' or 'All' in optimization_level:
		table['Optimization_Level'] = 'a.MMD_PL_Site IS NOT NULL' #this is a placeholder in the SQL for no constraint
	elif 'DSPs' in optimization_level:
		table['Optimization_Level'] = "a.MMD_PL_Site LIKE '%GoogleDBM%'"
	elif 'Direct' in optimization_level:
		table['Optimization_Level'] = "a.MMD_PL_Site NOT LIKE '%GoogleDBM%'"

	#Product Categories MDM_CA_Product_Categories IN e.g., 'Appliances,Major Appliances,Small Appliances,Connected Home'
	if ',' in str(data['Product_Categories']): # it's multiple channels in a list, i.e., [u'Video,Display']
		table['MDM_CA_Product_Categories'] = str(data['Product_Categories'][0])
	else:
		table['MDM_CA_Product_Categories'] = str(data['Product_Categories'])
	
	
	#Fiscal Year Selection; IN ('FY18'), or ('FY18','FY19')
	if ',' in str(data['Fiscal_Year_Selection']):
		table['Fiscal_Year_Selection'] = str(data['Fiscal_Year_Selection'][0])
	else:
		table['Fiscal_Year_Selection'] = str(data['Fiscal_Year_Selection'])
	
	#not used
	##table['Name_Optimization_Run'] = str(data['Name_Optimization_Run'])
	##table['Username'] = str(data['Username'])
	##table['Date'] = str(data['Date'])
	return table
	
def build_sql_query(p, channel_breakout=None): #parameters
	#Campaign Goal
	if p['MDM_CA_Campaign_Objective'] == 'Conversion':
		KPI_field = 'True_Online_Revenue'
	elif p['MDM_CA_Campaign_Objective'] == 'Awareness':
		KPI_field = 'Reach'
	elif p['MDM_CA_Campaign_Objective'] == 'Consideration':
		KPI_field = 'Clicks'
	
	#optimization level
	MDM_PL_Site = p['Optimization_Level']
		
	#Channels; need to change format for SQL query list, e.g., "Video,Display" -> "'Video','Display'"
	if channel_breakout is None:
		MDM_PL_Channels = "'" + p['MDM_PL_Channels'].replace(',','\',\'') + "'"
	else:
		MDM_PL_Channels = "'" + channel_breakout.replace(',','\',\'') + "'"
	
	#Campaign Names to restrict query to
	Campaign_Names = "'" + p['CampaignNames'].replace(',','\',\'') + "'"
	
	#Fiscal Year
	Fiscal_Year_Selection = "'" + p['Fiscal_Year_Selection'].replace(',','\',\'') + "'"
	
	#Vendor Names
	MDM_CA_Vendor_Names = "'" + p['MDM_CA_Vendor_Names'].replace(',','\',\'') + "'"
	
	#MDM_CA_Product_Categories
	MDM_CA_Product_Categories = "'" + p['MDM_CA_Product_Categories'].replace(',','\',\'') + "'"
	
	#MDM_CA_Campaign_Types
	MDM_CA_Campaign_Types = "'" + p['MDM_CA_Campaign_Types'].replace(',','\',\'') + "'" #updated 1/23/18
	
	#Date Range
	Campaign_Length_Range = p['Campaign_Length_Range']
	
	sql_query = \
	"SELECT CONCAT(a.Year,a.Week) AS [YearWeek], a.DCM_PlacementID, ROUND(" + KPI_field + ",2) AS KPI, ROUND(a.CappedCost,2) AS Weekly_Spend, a.PlacementName, \
	a.MDM_PL_Channel, a.Prisma_PackageName, a.CampaignName, a.Environment, a.Placement_Type, a.[Site], a.DSP_Data_Provider, a.Social_Post_Type, \
	a.Video_Content, a.Segment_Name, a.Segment_Type, a.Targeting_Type, a.Creative_size \
	FROM [DW_BestBuy].[Kaleidoscope].[Weekly_Placement_KaleidoscopeData] a \
	JOIN (\
	SELECT DCM_PlacementID, count(*) AS cnt \
	FROM [DW_BestBuy].[Kaleidoscope].[Weekly_Placement_KaleidoscopeData] \
	GROUP BY DCM_PlacementID \
	HAVING count(*) >= 2\
	) b \
	ON a.DCM_PlacementID = b.DCM_PlacementID \
	WHERE RIGHT(a.Fiscal_Year_Selection,4) IN (" + Fiscal_Year_Selection + ") AND \
	a.MDM_PL_Channel IN (" + MDM_PL_Channels + ") AND \
	a.MDM_CA_Product_Category IN (" + MDM_CA_Product_Categories + ") AND \
	a.CampaignName IN (" + Campaign_Names + ") AND \
	a.MDM_CA_Campaign_Type IN (" + MDM_CA_Campaign_Types + ") AND \
	a.MDM_CA_Vandor_Name IN (" + MDM_CA_Vendor_Names + ") AND " + \
	Campaign_Length_Range + " AND " + \
	MDM_PL_Site + " AND " + \
	KPI_field + " IS NOT NULL " + "\
	ORDER BY a.DCM_PlacementID, [YearWeek] ASC;"
	
	sql_query = sql_query.replace('\t','')
	return sql_query
	
def execute_sql(query, server_connection_string):
	if verbose == True: print 'SQL Query: ', query, '\n'
	cnxn = pyodbc.connect(server_connection_string)
	cursor = cnxn.cursor()
	cursor.execute(query)
	rows = cursor.fetchall()
	return rows

###CURVE RELATED FUNCTIONS START###
def parabola(x,a,b):
	return a*(x**2) + b*x

def _find_vertex(a,b):
	#dy/dx = 2ax + b = 0
	x = -b / (2*a)
	y = a * (x**2) + b * (x)
	return x,y
	
def neg_exp(x,a,b):
	return a*(1-np.exp(-b*x))
	
def preprocess_curves(data):
	d = D_list (dict)
	for line in data:
		YYYMM = line[0]
		placement_ID = line[1]
		KPI = line[2] #y val
		spend = line[3] #x val
		if spend == None: #NULL comes through as zero here; 1/29/18
			spend = 0
			KPI = 0 #no credit should be attributed to spend of zero
		try:
			d[placement_ID].append([float(spend),float(KPI)])
		except:
			if verbose == True: print "In preprocess_curves(), problem with line: ", line
	return d
	
def fit_curves(curve_datapoints,output_curve_path, verbose):
	fh_out = open(output_curve_path,'w')
	print >> fh_out, 'Placement_ID|alpha|beta|max_x_val|max_y_val|calc_type|n' #print header row
	d = D_list (dict)
	for placement_ID in curve_datapoints:
		if len(curve_datapoints[placement_ID]) >= 3:
			if verbose == True: print 'placement_ID:',placement_ID
			#calculate response curve coefficients, start with fitting a parabola
			x = [] #spend $
			y = [] #KPI
			max_x_val=0
			max_y_val=0
			for pair in curve_datapoints[placement_ID]:
				x.append(pair[0])
				y.append(pair[1])
				if pair[0] > max_x_val:
					max_x_val = pair[0]
				if pair[1] > max_y_val:
					max_y_val = pair[1]
			#count number of elements which are	0
			cnt=0
			for i in range(0,len(x)):
				cnt+=1
			#fit a parabola
			popt, pcov = curve_fit(parabola,x,y)
			a = popt[0]
			b = popt[1]
			if verbose == True: print 'parabola coefficients:',a,b
			#find the vertex by setting 0st derivative to zero
			alpha = ''
			alt_calc_flag = ''
			if a < 0 and cnt >=3: #normal case, with inverted parabola and with at least three data points
				(_,alpha) = _find_vertex(a,b)
				alt_calc_flag='0'
				if verbose == True: print 'max(y):', max(y)
				if verbose == True: print 'curve fit alpha:', alpha
				#constrain the asymptote to 120% of the max y value when the alpha is more than 120% of
				#the max y value, or if max(y) is greater than the alpha, bump the alpha up to max(y)
				if max(y) > alpha:
					alpha = max(y)
					alt_calc_flag = '1'
				elif alpha > max(y)*1.2:
					alpha = max(y)*1.2
					alt_calc_flag = '2'
			else: #use 120% of max for alpha > 0 and/or have limited historical data
				alpha = max(y)*1.2
				alt_calc_flag = '3'
			#fit negative exponential with a fixed alpha from the y value of the parabola vertex
			popt, pcov = curve_fit(lambda x, b: neg_exp(x,alpha,b),x,y,0)
			#alpha = popt[0]
			beta = popt[0]
			if verbose == True: print 'negative exponential coefficients:', alpha, beta,'\n'
			print >> fh_out, '%s|%s|%s|%s|%s|%s|%s' % (placement_ID,str(alpha),str(beta),max_x_val,max_y_val,alt_calc_flag,str(cnt))
			d[placement_ID].append([str(alpha),str(beta),max_x_val,max_y_val,alt_calc_flag,str(cnt)])
	fh_out.close()
	return d
###CURVE RELATED FUNCTIONS END###

###CONSTRAINED OPTIMIZATION RELATED FUNCTIONS START###
def create_objective_function(A,B): #dynamically created objective function with alphas, betas; MIN -Y = SUM [a*(1-e^(-b*x))]
	objective = ""
	parameters = ""
	for i in range(0,len(A)):
		objective+= A[i] + "*(1-np.exp(-" + B[i] + "*c"+ str(i+1) + ")) + "
		parameters+= "c" + str(i+1) + ", "
	objective = objective[0:len(objective)-3]
	parameters = parameters[0:len(parameters)-2]
	if verbose == True: print 'Objective Function: ', objective, '\n'
	obj_func = """def custom_function(params):
		%s = params
		return -(%s)""" #minimize the negative
	exec(obj_func % (parameters, objective))
	return custom_function

def create_budget_constraint(nbr_of_placements, weekly_budget):
	eq = ""
	for i in range(0,nbr_of_placements):
		eq+= "c[" + str(i) + "] + "
	eq = eq[0:len(eq)-3]
	eq += " - " + str(weekly_budget)
	if verbose == True: print "Budget Constraint: " + eq + "\n"
	custom_constraint = """def budget_constraint(c):
		return %s"""
	exec(custom_constraint % (eq))
	return budget_constraint

def optimize_plan(curves, budget, weeks, log_file_path, placement_budget_limits, error_file_path, verbose):
	A = []
	B = []
	IDs = []
	fh_out = open(log_file_path,'a')
	for placement_ID in curves:
		#print "placement_ID: ", placement_ID
		#print curves[placement_ID]
		alpha = curves[placement_ID][0][0]
		beta = curves[placement_ID][0][1]
		try:
			if float(beta) > 0: #added 1/31/18, and indented three lines below, as a test to speed up script
				A.append(alpha)
				B.append(beta)
				IDs.append(placement_ID)
		except:
			if verbose == True: print 'problem in optimize_plan() with the following row in the curves input: ', curves[placement_ID]
			pass
	objective_function = create_objective_function(A,B)
	z = 0 # initial condition for each budget placement in iterative optimization
	C = [z] * len(A) #initial conditions
	weekly_budget = str(budget/weeks)
	nbr_of_placements = len(IDs)
	budget_constraint = create_budget_constraint(nbr_of_placements, weekly_budget)
	cons = {'type':'eq', 'fun': budget_constraint}
	
	if placement_budget_limits == True and nbr_of_placements < 1000: #added 1/29/18
		bnds = ""
		for placement_ID in IDs: #1/31/18 changed from curves to IDs to simplify obj func removing curves with beta = 0
			try:
				max_x_val = float(curves[placement_ID][0][2]) #never recommend more than 2x over previous max
				bnd = "(0, " + str(max_x_val * 2) + "),"
			except:
				bnd = "(0, None),"
			bnds+=bnd
	else: #False
		bnd = "(0, None),"
		bnds = bnd * len(A)
	if verbose == True: print 'Placement Bounds: ', bnds, '\n'
	bnds = eval("(" + bnds[0:len(bnds)-1] + ")")
	
	try:
		if nbr_of_placements < 500: #2/5/18
			result = minimize(fun=objective_function, x0=C, method='SLSQP', constraints=cons, bounds=bnds, options={'ftol': 1e-4, 'disp': True, 'maxiter':1000})
		else:
			result = minimize(fun=objective_function, x0=C, method='SLSQP', constraints=cons, bounds=bnds, options={'ftol': 1e-2, 'disp': True, 'maxiter':500})
	except Exception as e:
		print >> fh_out, "The optimizer experienced a fatal error: " + str(e)
		fh_out.close()
		fh_err = open(error_file_path,'w')
		print >> fh_err, "ERROR: The optimizer was unable to arrive at a solution. Try broadening the query and/or changing the budget. The specific error was: '" + str(e) + "'"
		fh_err.close()
		raise
	
	if result.success == True or (result.success == False and 'teration' in result.message and 'nan' not in str(result.fun)): #[I,i]teration[s]
		weekly_spend_allocations_by_placement = result.x
		print >> fh_out, result
		if verbose == True:	print(result)
	else:
		#print 'result.message: ', result.message
		if verbose == True: print "A fatal error occurred. The optimizer was unable to arrive at a solution. See the log.\n\n" + str(result)
		
		if str(result.fun) == 'nan': #budget likely exceeded 2x the sum of the maximum highest historical placement spends
			print >> fh_out, "The optimizer was unable to arrive at a solution. The budget likely exceeds 2x historical maximum spends across placements. Try reducing the budget amount.\n\n" + str(result)
			fh_err = open(error_file_path,'w')
			print >> fh_err, "ERROR: The optimizer was unable to find a solution. Try reducing the budget or increasing the number of included placements."
			fh_err.close()
			weekly_spend_allocations_by_placement = result.x #np.array(C)
			fh_out.close()
		else:
			print >> fh_out, "The optimizer was unable to arrive at a solution.\n\n" + str(result)
			fh_err = open(error_file_path,'w')
			print >> fh_err, "ERROR: The optimizer was unable to arrive at a solution. Try simplifying the plan."
			fh_err.close()
			weekly_spend_allocations_by_placement = result.x #np.array(C)
			fh_out.close()
		raise ValueError(result.message)
	fh_out.close()
	plan = [] #added sort descending on spend 2/2/18
	for i in range(0,len(IDs)):
		plan.append([IDs[i],round(weekly_spend_allocations_by_placement[i],2)])
	plan.sort(key=lambda k: (k[1]), reverse=True) #sort descending on spend
	return plan
###CONSTRAINED OPTIMIZATION RELATED FUNCTIONS END###
	
def run_optimization(parameters, working_directory_path, server_connection_string, placement_budget_limits, verbose, breakout_offset=0):

	if parameters['Optimization_Type'].split(' ')[0] == 'Open':
		curve_fn = 'All.txt'
		plan_fn = 'report.csv'
		budget =  float(parameters['Optimization_Budget'].split(',')[0]) #should be only one number, but this is safe
		log_file_path = working_directory_path + 'log.txt'
		sql = build_sql_query(parameters)
	else: # parameters['Optimization_Type'].split(' ')[0] == 'Breakout':
		current_channel = parameters['MDM_PL_Channels'].split(',')[int(breakout_offset)]
		curve_fn = current_channel + '.txt'
		budget =  float(parameters['Optimization_Budget'].split(',')[int(breakout_offset)])
		plan_fn = 'report_' + current_channel + '.csv'
		log_file_path = working_directory_path + 'log_' + current_channel + '.txt'
		sql = build_sql_query(parameters, current_channel)
	
	output_plan_path = working_directory_path + plan_fn
	output_curve_path = working_directory_path + curve_fn
	error_file_path = working_directory_path + 'errors.txt'
	
	fh_out = open(log_file_path,'w')
	#reformat SQL so easier to read in log file
	psql = sql
	psql = psql.replace(" FROM","\nFROM")
	psql = psql.replace(" WHERE","\nWHERE")
	psql = psql.replace(" JOIN","\nJOIN")
	psql = psql.replace(" ON","\nON")
	psql = psql.replace(" GROUP BY","\nGROUP BY")
	psql = psql.replace(" HAVING","\nHAVING")
	psql = psql.replace(" AND","\nAND")
	psql = psql.replace(" ORDER BY","\nORDER BY")
	print >> fh_out, "SQL Query:\n" + str(psql)
	fh_out.close()

	dataset=[]
	try:
		dataset = execute_sql(sql, server_connection_string)
	except:
		if verbose == True: print 'SQL query failed to execute\n'
		fh_err = open(error_file_path,'w')
		print >> fh_err, 'ERROR: SQL query failed to execute. Could not connect to server.'
		fh_err.close()
		raise ValueError('SQL query failed to execute')
	if dataset == []:
		if verbose == True: print 'SQL query returned no results. Re-specify the query to be broader and rerun.\n'
		fh_err = open(error_file_path,'w')
		print >> fh_err, 'ERROR: SQL query returned no results. Respecify the query to be broader and rerun.'
		fh_err.close()
		raise ValueError('SQL query returned no result')
		
	curve_datapoints = preprocess_curves(dataset)
	curves = fit_curves(curve_datapoints, output_curve_path, verbose)
	weeks = int(data['Campaign_Length'][:2])
	
	plan = optimize_plan(curves,budget,weeks,log_file_path,placement_budget_limits,error_file_path,verbose)

	#create {placementID: [PlacementName, MDM_PL_Channel, Prisma_PackageName]} lookup dictionary
	#2/26/18 added CampaignName, Environment, Placement_Type, Site, DSP_Data_Provider, Social_Post_Type, \
	#  Video_Content, Segment_Name, Segment_Type, Targeting_Type, Creative_Size
	p = D_list (dict)
	for placement in dataset:
		p[placement[1]] = [placement[5],placement[4],placement[6],placement[7],placement[8],placement[9],placement[10],placement[11],placement[12],placement[13],placement[14],placement[15],placement[16],placement[17]]

	#save results of plan
	fh_out = open(output_plan_path,'w')
	print >> fh_out, '"Channel","Placement_ID","Package","Placement_Name","Recommended_Total_Spend","CampaignName","Environment","Placement_Type","Site","DSP_Data_Provider","Social_Post_Type","Video_Content","Segment_Name","Segment_Type","Targeting_Type","Creative_Size"'
	for i in range(0,len(plan)):
		ID = long(plan[i][0])
		spend = float(plan[i][1]) * weeks #this scales the placement budgets from weekly back up to the full length of a campaign
		print >> fh_out, '"' + str(p[ID][0]) + '","' + str(ID) + '","' +  str(p[ID][2]) + '","' + str(p[ID][1]) + '","' + str(round(spend,2)) + '","' +  str(p[ID][3]) + '","' +  str(p[ID][4])+ '","' +  str(p[ID][5])+ '","' +  str(p[ID][6])+ '","' +  str(p[ID][7])+ '","' +  str(p[ID][8])+ '","' +  str(p[ID][9])+ '","' +  str(p[ID][10])+ '","' +  str(p[ID][11])+ '","' +  str(p[ID][12])+ '","' +  str(p[ID][13]) + '"'
	fh_out.close()

	#need to append spend back to curves
	fh_out = open(output_curve_path,'w') #overwrite, add spend column, and sorted by spend desc
	print >> fh_out, 'Placement_ID|placement_name|alpha|beta|max_x_val|max_y_val|recommended_weekly_spend|calc_type|n' #print header row
	for i in range(0,len(plan)):
		ID = long(plan[i][0])
		spend = float(plan[i][1]) #WEEKLY recommended spend, note, not the full plan
		placement_name = p[ID][1] #placement name
		alpha = str(curves[ID][0][0])
		beta = str(curves[ID][0][1])
		max_x_val = curves[ID][0][2]
		max_y_val = curves[ID][0][3]
		alt_calc_flag = curves[ID][0][4]
		cnt = str(curves[ID][0][5])
		print >> fh_out, '%s|%s|%s|%s|%s|%s|%s|%s|%s' % (ID,placement_name,alpha,beta,max_x_val,max_y_val,spend,alt_calc_flag,cnt)
	fh_out.close()
	#END OF RUN_OPTIMIZATION FUNCTION

##########################################################
#STEPS
data = read_json(working_directory_path,verbose)
parameters = return_sql_parameters(data)
optimization_type = parameters['Optimization_Type'].split(' ')[0]
total_breakouts = len(parameters['MDM_PL_Channels'].split(','))

if optimization_type == 'Open':
	run_optimization(parameters, working_directory_path, server_connection_string, placement_budget_limits, verbose)
	log_file_path = working_directory_path + 'log.txt'
else: #do a breakout optimization
	for x in range(0,total_breakouts):
		run_optimization(parameters, working_directory_path, server_connection_string, placement_budget_limits, verbose, breakout_offset=x)
		log_file_path = working_directory_path + 'log_' + parameters['MDM_PL_Channels'].split(',')[x] + '.txt'

#zip plan files for download
os.chdir(working_directory_path)
for file in os.listdir(working_directory_path):
    if file.endswith(".csv"):
		zipfile.ZipFile('report.zip', mode='a').write(file)
		os.remove(file)

#end of script actions
end = t.time()
elapsed = end - start

if verbose == True:
	print "done creating response curves"
	print "Run time: ", elapsed, " seconds\n"

fh_out = open(log_file_path,'a')
print >> fh_out, "Run time: ", elapsed, " seconds\n"
fh_out.close()
