import matplotlib.pyplot as plt
import config_reader as config_reader
import numpy as np

val_path = config_reader.load_val_path()
val_data = []
with open(val_path) as f:
    lines = f.readlines()
    for line in lines:
        val_data.append(float(line.split(':')[1]))

print('max acc {} in epoch {}'.format(np.max(val_data), np.argmax(val_data)))
plt.plot(val_data)
plt.xlabel('epoch')
plt.ylabel('val_acc')
plt.show()