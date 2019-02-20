
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint
import datetime
class LossHistory(Callback):

    def __init__(self,run,learning_curve):
        self.run = run
        self.i = 0
        self.learning_curve = learning_curve
        
    def on_epoch_end(self,batch,logs):
        self.run.log_scalar("training.loss",
                            logs.get('loss'),
                            self.i)

        self.run.log_scalar("validation.loss",
                            logs.get('val_loss'),
                            self.i)

#        self.learning_curve[self.i] = logs.get('val_loss')
        self.i = self.i + 1

def fetch_callbacks(options,_run,learning_curve):
    """
    A refugee center for the boilerplate callback codes

    Parameters:
    -----------
    options["noise"] - noise level concetanated as an exp string
    options["experiment"] - model name
    options["k"] - current fold
    _run - Sacred experiment run variable
    learning_curve - learning curve array updated pass by reference
    Returns:
    --------
    Callbacks array for the Keras model
    """

    date_at_start = datetime.datetime.now()
    date_string = date_at_start.strftime("%y-%b-%d-%H-%m")

    tb = TensorBoard(log_dir="logs/" +
                     date_string +
                     "_" +
                     str(options["noise"]))


    mc = ModelCheckpoint("checkpoints/" +
                         options["experiment"] +
                         str(options["k"]) +
                         ".hdf5",
                         save_best_only=True)

    lh = LossHistory(_run,learning_curve)

    return [tb,mc,lh]

def fetch_callbacks_norun(options):
    """
    A refugee center for the boilerplate callback codes
    which does not inclue the run argument
    Parameters:
    -----------
    options["noise"] - noise level concetanated as an exp string
    options["experiment"] - model name
    options["k"] - current fold
    _run - Sacred experiment run variable
    learning_curve - learning curve array updated pass by reference
    Returns:
    --------
    Callbacks array for the Keras model
    """

    date_at_start = datetime.datetime.now()
    date_string = date_at_start.strftime("%y-%b-%d-%H-%m")

    mc = ModelCheckpoint("checkpoints/" +
                         options["experiment"] +
                         str(options["k"]) +
                         ".hdf5",
                         save_best_only=True)

    return [mc]
