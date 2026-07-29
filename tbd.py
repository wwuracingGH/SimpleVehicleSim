with open('defaulttrack.csv') as f:
    ldat = f.readline().split(',')
    while line := f.readline():
        dat = line.split(',')
        print(f"[({ldat[4]},{ldat[5]},0),({dat[0]},{dat[1]},0),({dat[2]},{dat[3]},0)],", end='')
        ldat = dat
        