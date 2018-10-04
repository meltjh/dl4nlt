import argparse 
import matplotlib.pyplot as plt
import numpy as np 
plt.rcParams.update({'font.size': 12})
parser = argparse.ArgumentParser()
# FOR EXAMPLE RUNNING COMMAND: 
# python plot_val.py --dir_val val_lr0.001_batchsize64_embeddim50_hidden256_layers1.txt val_lr1e-05_batchsize64_embeddim50_hidden256_layers1.txt validation_lr1e-4-256.txt --lr 1e-03 1e-04 1e-05 --output 'val_256_lr.png'
# DUMMY COMMAND 
# python plot_val.py --dir_val TXTFILE1.txt TXTFILE2.txt --lr LR1 LR2 --output 'plot.png'

parser.add_argument('--dir_val', nargs='+', help='specify each validation accuracy results text file')
parser.add_argument('--lr',nargs='+', help='values of learning rate respectively')
parser.add_argument('--output', type=str, help='name of output plot .png')
config, unparsed = parser.parse_known_args()

# file_names = config.dir_val.split(",")
file_names = config.dir_val
print(config.lr[0])
print(file_names)
for idx, file in enumerate(file_names):
	with open(file, 'r') as f:
		accuracy = [float(line.split()[0]) for line in f]
		xi = np.arange(0, len(accuracy), 2)
		plt.plot(accuracy , label=config.lr[idx])
		plt.xticks(xi)
		# plt.axis(np.arange(1,len(accuracy)+1))
plt.legend()
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.savefig(config.output)
plt.show()

