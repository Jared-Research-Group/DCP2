from core_scripts.lembox import getLemboxData
import matplotlib.pyplot as plt

def getData():
    data = []

    data.append(getLemboxData("D:\meeting\\testing\Layer 48\data_collection_20251007_082532\lembox_data.csv"))
    data.append(getLemboxData("D:\meeting\\testing\Layer 51\data_collection_20251007_093235\lembox_data.csv"))
    data.append(getLemboxData("D:\meeting\\testing\Layer 54\data_collection_20251007_113323\lembox_data.csv"))
    data.append(getLemboxData("D:\meeting\\testing\Layer 57\data_collection_20251007_124306\lembox_data.csv"))
    data.append(getLemboxData("D:\meeting\\testing\Layer 61\data_collection_20251007_141801\lembox_data.csv"))

    return data

def plotData(d):

    fig, ax = plt.subplots(len(d),2, constrained_layout=True)

    ax[0][0].set_title('Average Current')
    ax[0][1].set_title('Average Voltage')

    ax[-1][0].set_xlabel('Time (s)')
    ax[-1][1].set_xlabel('Time (s)')

    for i, data in enumerate(d):
        ax[i][0].scatter(data[0],data[3])
        ax[i][1].scatter(data[0],data[4])

        ax[i][0].set_ylabel('Current (A)')
        ax[i][1].set_xlabel('Voltage (V)')

    plt.show()

def main():

    dat = getData()

    plotData(dat)

if __name__ == '__main__': main()