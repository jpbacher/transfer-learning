from poutyne.framework.callbacks import Callback


class CallbackComet(Callback):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch_number, logs):
        self.experiment.log_metrics(logs)
