
import datetime
from keras import optimizers

def path_name_inf(options, logged_options,path_name):
    for option in options:
        if (option in logged_options):
            path_name = path_name + "_" + option + "_" +  str(options[option])
    path_name = path_name + datetime.datetime.now().strftime("%b-%d-%H:%M")
    return path_name

def optimiser_handler(options):
    if (options["optimiser"] == "adam"):
        # Tuning recommendation: learning rate
        optimiser = optimizers.Adam(lr=options["lr"])
        
    if (options["optimiser"] == "rmsprop"):
        # Tuning recommendation: learning rate
        optimiser = optimizers.RMSprop(lr=options["lr"])

    if (options["optimiser"] == "sgd"):
        # Tuning recommendation: Basically everything

        if "momentum" not in options.keys():
            options["momentum"] = 0
        if "nesterov" not in options.keys():
            options["nesterov"] = False
        
        if "clipnorm" in options.keys():
            optimiser = optimizers.SGD(lr=options["lr"], clipnorm=options["clipnorm"])
        else:
            optimiser = optimizers.SGD(lr=options["lr"], momentum = options["momentum"])
            
        
    return optimiser

def experiment_logger(options):
    file_obj = open("logs/total.txt","a")
    file_obj.write("date:" + datetime.datetime.now().strftime("%b-%d-%H:%M") + ",")
    for option in options:
        file_obj.write(option + ":" + str(options[option]) + ",")
    file_obj.write("\n")
    file_obj.close()
