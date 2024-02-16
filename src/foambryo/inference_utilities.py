import numpy as np

def open_pressure_file(filename,ncells): 
    File = open(filename, 'r+')
    File.readline()
    pressure=[]
    for i in range(ncells):
        File.readline()
        pressure.append(float(File.readline()[:-1]))
        
    return(np.array(pressure)/6)

def open_pressure_dict(filename,ncells): 
    File = open(filename, 'r+')
    File.readline()
    pressure=[]
    for i in range(ncells):
        File.readline()
        pressure.append(float(File.readline()[:-1]))
        
    dict_pressure = {0:0}
    for i in range(ncells): 
        dict_pressure[i+1]=pressure[i]
    return(dict_pressure)

def open_tensions_dict(filename,dict_tensions,target_tension=1): 
    #dict_tensions is the dictionnary of infered tensions
    cells_info = np.load(filename,allow_pickle=True).item()
    
    dict_gt ={}
    for key in sorted(dict_tensions.keys()): 
        if key[0]==0 : 
            dict_gt[key] = cells_info['Gamma'][key[1]-1]
        else : 
            dict_gt[key] = cells_info['YIJ'][key[0]-1,key[1]-1]
    m = sum(dict_gt.values())/len(dict_gt)
    for key in dict_gt.keys(): 
        dict_gt[key]/=(m/target_tension)
    return(dict_gt,m/target_tension)

def renormalize_pressures(dict_pressure,factor): 
    #All the tensions are renormalized to have for mean 1. 
    #We need to normalize the pressures by the same coefficient to be consistent. 
    for key in dict_pressure.keys(): 
        dict_pressure[key]/=factor
    return(dict_pressure)

def open_pressure_dict_renormalized(filename, ncells,factor): 
    dict_pressure = open_pressure_dict(filename,ncells)
    return(renormalize_pressures(dict_pressure,factor))

def loss_l2(d1,d2): 
    #compute the loss between all the elements of two dictionnaries assuming both dictionnaries have the same keys
    loss=0
    n=0
    for key in d1.keys(): 
        if not(key in d2.keys()): 
            continue
        n+=1
        loss+=(d1[key]-d2[key])**2
    return(loss/n)

def loss_l1(d1,d2): 
    #compute the loss between all the elements of two dictionnaries assuming both dictionnaries have the same keys
    loss=0
    n=0
    for key in d1.keys(): 
        if not(key in d2.keys()): 
            continue
        n+=1
        loss+=np.abs(d1[key]-d2[key])
        
    return(loss/n)

def loss_ratio_l2(d1,d2): 
    #compute the loss between all the elements of two dictionnaries assuming both dictionnaries have the same keys
    #d2 is the ground truth by convention
    #print("Make sure that you put the ground truth as second argument")
    losses = []
    for key in d1.keys(): 
        if not(key in d2.keys()): 
            continue
        if key == 0 : continue
        losses.append(((d1[key]-d2[key])/d2[key])**2)
    losses = np.array(losses)
    return(np.mean(losses))


def loss_ratio_l1(d1,d2): 
    #compute the loss between all the elements of two dictionnaries assuming both dictionnaries have the same keys
    #d2 is the ground truth by convention
    #print("Make sure that you put the ground truth as second argument")
    losses = []
    for key in d1.keys(): 
        if not(key in d2.keys()): 
            continue
        if key == 0 : continue
        losses.append(np.abs((d1[key]-d2[key])/d1[key]))
    return(np.mean(losses))

def loss_l2_normalized(d1,d2): 
    #compute the loss between all the elements of two dictionnaries assuming both dictionnaries have the same keys
    #d2 is the ground truth by convention
    #print("Make sure that you put the ground truth as second argument")
    losses = []
    mean_d2 = np.mean(list(d2.values()))
    for key in d1.keys(): 
        if key == 0 : continue
        losses.append(((d1[key]-d2[key])/mean_d2)**2)
    losses = np.array(losses)
    return(losses, np.mean(losses))


def loss_l1_normalized(d1,d2): 
    #compute the loss between all the elements of two dictionnaries assuming both dictionnaries have the same keys
    #d2 is the ground truth by convention
    #print("Make sure that you put the ground truth as second argument")
    losses = []
    mean_d2 = np.mean(list(d2.values()))
    for key in d1.keys(): 
        
        if key == 0 : continue
        losses.append(np.abs((d1[key]-d2[key])/mean_d2))
    return(losses, np.mean(losses))

