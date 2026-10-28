import zarr
import helper_functions
import sensors
from pathlib import Path

sensor_list = (sensors.LEMBox, sensors.Microphone, sensors.ThermocoupleDAQ, sensors.RSI, sensors.FLIR, sensors.Xiris)

def backcompatiblity_constructor(dir = None):

    if dir is None:
        dir = helper_functions.selectFolder()

    zarr.create_group(dir / (dir.name + '.zarr'))

    for file in dir.iterdir():
        sensor = None

        if file.stem == 'lembox_data':
            sensor = sensors.LEMBox(file)

        if file.stem == 'thermocouple_data':
            sensor = sensors.ThermocoupleDAQ(file)

        if file.stem == 'microphone_data':
            sensor = sensors.Microphone(file)

        if file.stem == 'robot_data':
            sensor = sensors.RSI(file)

        if file.name == 'FLIR':
            sensor = sensors.FLIR(file)

        if file.name == 'Xiris':
            sensor = sensors.Xiris(file)

        del sensor

if __name__ == '__main__':

    dir = helper_functions.selectFolder()

    if not dir.name.endswith('.zarr'):
        backcompatiblity_constructor(dir)

        dir = Path(str(dir) + '.zarr')

    root = zarr.open_group(dir)

    if not 'derived' in root:
        root.create_group('derived')

    lem = sensors.LEMBox(dir)
    lem.detect_welding(flag_plot = True)

    rsi = sensors.RSI(dir)
    rsi.detect_welding(flag_plot = True)

    mic = sensors.Microphone(dir)
    mic.detect_welding(flag_plot = True)