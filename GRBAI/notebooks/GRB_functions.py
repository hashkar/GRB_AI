####################
#This file contains low level usefull funcitons  for GRB data analysis
####################

#importing required packages
import pandas as pd
import swifttools.ukssdc.data.GRB as udg
import numpy as np
#importing required packages from astropy and gammapy 


#function to extract XRT data from data on GRB
def get_grb_xrt_data(TriggerID) -> pd.DataFrame:
    ''' Returns XRT data frame'''

    if type(TriggerID) is not int:
        raise TypeError("GRB should be an integer.")
 
    try:
        data = udg.getBurstAnalyser(targetID = TriggerID,
                                    instruments=('XRT',),
                                    saveData=False,
                                    returnData=True)
    except:
        xrt_dataframe = pd.DataFrame(columns=["TrigID", "NameGRB", "Time", "TimeNeg", "TimePos", "Flux", "FluxNeg", "FluxPos"])
        return xrt_dataframe

    # raise exception if no GRB is found
    if len(data['Instruments']) == 0:
        #raise ValueError(f"No GRBs found with trigger ID {triggerID}")
        xrt_dataframe = pd.DataFrame(columns=["TrigID", "NameGRB", "Time", "TimeNeg", "TimePos", "Flux", "FluxNeg", "FluxPos"])
        return xrt_dataframe
    # create final dataframe and adding XRT data if available
    xrt_dataframe = pd.DataFrame(columns=["TrigID", "NameGRB", "Time", "TimeNeg", "TimePos", "Flux", "FluxNeg", "FluxPos"])
    xrt_df_list = [xrt_dataframe]
    if "XRTBand_WT_incbad" in data["XRT"]["Datasets"]:
        xrt_wt = data["XRT"]["XRTBand_WT_incbad"][["Time", "TimeNeg", "TimePos", "Flux", "FluxNeg", "FluxPos"]]
        xrt_df_list.append(xrt_wt)
    if "XRTBand_PC_incbad" in data["XRT"]["Datasets"]:
        xrt_pc = data["XRT"]["XRTBand_PC_incbad"][["Time", "TimeNeg", "TimePos", "Flux", "FluxNeg", "FluxPos"]]
        xrt_df_list.append(xrt_pc)

    if len(xrt_df_list) == 1:
        raise ValueError(f"No XRT data found for GRB with trigger ID {triggerID}")

    xrt_dataframe = pd.concat(xrt_df_list)
    xrt_dataframe = xrt_dataframe.assign(TrigID=TriggerID)
    xrt_dataframe.reset_index(drop=True, inplace=True)

    return xrt_dataframe


# Function to calculate the power-law with constants a and b using log-values
def log_power_law(x, a, b):
    ''' Returns a logarithmic power law function'''
    return b*x+np.log(a)

def isfloat(number):
    try:
        float(number)
        return True
    except ValueError:
        return False
    

def associate_GRB_to_redshift(swift_id, name_dict, name_data, redshift_data):
    ''' Returns the redshift of a GRB and stores the GRB name in the name_dict dictionary'''
    
    exceptions = [1121751, 954304] #, 775946, 1004219, 1076121, 1078701, 1082569, 1086826, 596958, 603488]
    
    if swift_id in exceptions:
        return 'not in Swift BAT data', 'not in Swift BAT data'
        
    else:
        GRB_redshift = {} # dictionary whose keys and values are GRB swift_id and associated GRB name
        red = {} # dictionary whose keys and values are GRB name and associated redshift
        

        for line in name_data:
            number = line[1].strip()
            if np.char.isnumeric(number) == False:
                continue
            number_id = int(number)
            if number_id == swift_id:
                GRB_redshift[swift_id] = line[0].strip() 
    

        for line in redshift_data:
            name = line[0].strip()
            redshift = line[1].strip()
            red[name] = redshift
        try:
            print(GRB_redshift[swift_id])
            name_dict[swift_id] = GRB_redshift[swift_id]
            GRBNAME = str(name_dict[swift_id])
        except: 
            return 'not in Swift BAT data', 'not in Swift BAT data'
        if GRB_redshift[swift_id] in red and isfloat(red[GRB_redshift[swift_id]]) == True:
            return float(red[GRB_redshift[swift_id]]), GRBNAME
        else:
            return 'unknown redshift', 'unknown redshift'
