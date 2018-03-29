import os

LINE_TRAIN = os.path.expanduser('~/graph-trend-understanding/data/train_labels_linear.txt')
LINE_VAL = os.path.expanduser('~/graph-trend-understanding/data/val_labels_linear.txt')

SIN_TRAIN = os.path.expanduser('~/graph-trend-understanding/data/train_labels_sinusoid.txt')
SIN_VAL = os.path.expanduser('~/graph-trend-understanding/data/val_labels_sinusoid.txt')

EXP_TRAIN = os.path.expanduser('~/graph-trend-understanding/data/train_labels_exponential.txt')
EXP_VAL = os.path.expanduser('~/graph-trend-understanding/data/val_labels_exponential.txt')

FINAL_TRAIN = os.path.expanduser('~/graph-trend-understanding/data/train_labels.txt')
FINAL_VAL = os.path.expanduser('~/graph-trend-understanding/data/val_labels.txt')

def read(filename): 
    lines = []
    with open(filename, 'r') as infile: 
        for line in infile: 
            lines.append(line)
    return lines

def make_new_train(): 
    linetrain = read(LINE_TRAIN)
    print("len linetrain", len(linetrain))
    sintrain = read(SIN_TRAIN)
    print("len sintrain", len(sintrain))
    exptrain = read(EXP_TRAIN)
    print("len exptrain", len(exptrain))
    
    newtrain = linetrain[:25000] + sintrain + exptrain
    print("len newtrain", len(newtrain))
    
    with open(FINAL_TRAIN, 'w') as outfile: 
        for line in newtrain: 
            outfile.write(line)

def make_new_val(): 
    linetrain = read(LINE_VAL)
    print("len linetrain", len(linetrain))
    sintrain = read(SIN_VAL)
    print("len sintrain", len(sintrain))
    exptrain = read(EXP_VAL)
    print("len exptrain", len(exptrain))
    
    newtrain = linetrain[:2500] + sintrain + exptrain
    print("len newtrain", len(newtrain))
    
    with open(FINAL_VAL, 'w') as outfile: 
        for line in newtrain: 
            outfile.write(line)

if __name__ == '__main__':
    make_new_train()
    make_new_val()
















