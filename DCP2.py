import zarr
import helper_functions
import sensors

def backcompatiblity(dir = None):

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

    if not dir.endswith('.zarr'):
        backcompatiblity(dir)