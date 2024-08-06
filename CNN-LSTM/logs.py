import os
from datetime import datetime
from pathlib import Path

def logfile(hyperparameters, parameters, constants, **kwargs):
    
    
    kwargs["exp_date_text"] = datetime.now().strftime("%Y_%m_%d_%H_%M")
    theExpName = kwargs.get("exp_date_text") + "_Freq_" + str(kwargs.get("freq_flag")) + "_duration_" + str(kwargs.get("signaltime")) + "_Ep_" + str(kwargs.get("n_epochs"))+ "_lr_" + str(kwargs.get("lr"))+ "_dropout_" + str(kwargs.get("dropout_rate"))+ "_dropout_" + str(kwargs.get("model_select")[0])
    
    s_path = kwargs.get('working_directory') + "/exps/EXP{}"
    # this line is not redundant as the constants are required in the below with-open block
    constants["save_path"] =  os.path.realpath(Path(s_path.format(theExpName)))
    # update kwargs
    kwargs["save_path"] = constants["save_path"]
    
    if not(os.path.isdir("./exps/EXP{}".format(theExpName))):
            os.mkdir("./exps/EXP{}".format(theExpName))
            os.mkdir("./exps/EXP{}/figs".format(theExpName))
    
    logs_path = os.path.join(constants['save_path'], 'logs.txt')
    kwargs["logs_path"] = logs_path
    
    logs_path_Vs30_train = os.path.join(constants['save_path'], 'logs_Vs30_train.txt')
    kwargs["logs_path_Vs30_train"] = logs_path_Vs30_train
    
    with open(logs_path_Vs30_train, "w") as file:
        file.close()
        
    logs_path_Vs30_val = os.path.join(constants['save_path'], 'logs_Vs30_val.txt')
    kwargs["logs_path_Vs30_val"] = logs_path_Vs30_val
    
    with open(logs_path_Vs30_val, "w") as file:
        file.close()
        
    logs_path_Vs30_test = os.path.join(constants['save_path'], 'logs_Vs30_test.txt')
    kwargs["logs_path_Vs30_test"] = logs_path_Vs30_test
    
    with open(logs_path_Vs30_test, "w") as file:
            file.close()
        
    with open(logs_path, "w") as file:
    
        # Hyperparameters 
        file.write("Hyperparameters:\n")
        file.write("---------------\n")
        for key, value in hyperparameters.items():
            file.write("{}: {}\n".format(key, value))
        file.write("\n")
    
        # Parameters 
        file.write("Parameters:\n")
        file.write("-----------\n")
        for key, value in parameters.items():
            file.write("{}: {}\n".format(key, value))
        file.write("\n")
    
        # Constants 
        file.write("Constants:\n")
        file.write("----------\n")
        for key, value in constants.items():
            file.write("{}: {}\n".format(key, value))
        file.write("\n")
        file.close()
        
    return kwargs
        
