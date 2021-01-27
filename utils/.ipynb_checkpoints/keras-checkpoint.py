import matplotlib.pyplot as plt


def metric_plot(history,metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric])
    plt.title('model '+metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def history_plot(history):
    for metric in history.history:
        if not metric.startswith('val'):
            metric_plot(history,metric)