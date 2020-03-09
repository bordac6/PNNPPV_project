import matplotlib.pyplot as plt
import config_reader as config_reader
import numpy as np

val_path = config_reader.load_val_path()
val_data = []
val_data05 = []
with open(val_path) as f:
    lines = f.readlines()
    for line_idx in range(1, len(lines)):
        line = lines[line_idx - 1]
        line2 = lines[line_idx]
        if len(line) < 50:
            continue
        if len(line2) < 50:
            line += line2[:-1]
        val_data.append(float(line.split(':')[1]))
        val_data05.append(float(line.split(':')[-2]))
        try:
            if int(line2.split(':')[0].split(' ')[-1]) == 0:
                print('max acc {} in epoch {}'.format(np.max(val_data), np.argmax(val_data)))
                print('best {}'.format(val_data05[np.argmax(val_data05)]))
                plt.plot(val_data)
                plt.plot(val_data05)
                plt.xlabel('epoch')
                plt.ylabel('val_acc')
                plt.show()

                val_data = []
                val_data05 = []
                input()
        except:
            pass