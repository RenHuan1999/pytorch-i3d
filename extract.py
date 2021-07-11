import os

length = 1500
for start in range(0, 4500-1, length):
    os.system('python extract_features.py -start {} -length {} -gpu 0 >> logs_crop/extract_{}.log 2>&1 &'.format(start, length, start))
for start in range(4500, 9000-1, length):
    os.system('python extract_features.py -start {} -length {} -gpu 1 >> logs_crop/extract_{}.log 2>&1 &'.format(start, length, start))
for start in range(9000, 13500, length):
    os.system('python extract_features.py -start {} -length {} -gpu 2 >> logs_crop/extract_{}.log 2>&1 &'.format(start, length, start))
for start in range(13500, 17000, length):
    os.system('python extract_features.py -start {} -length {} -gpu 3 >> logs_crop/extract_{}.log 2>&1 &'.format(start, length, start))

