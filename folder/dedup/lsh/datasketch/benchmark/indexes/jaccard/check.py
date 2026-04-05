def rf(f):
    with open(f, 'r') as fi:
        return fi.readlines()
    
fp1 = set(rf("fp.txt"))
fpqi = set(rf("fp_qi.txt"))
fpqi2 = set(rf("fp_qi2.txt"))

print(len(fp1), len(fpqi), len(fpqi2))
print(len(fp1 & fpqi), len(fpqi & fpqi2), len(fp1 & fpqi2))

fn1 = set(rf("fn.txt"))
fnqi = set(rf("fn_qi.txt"))
fnqi2 = set(rf("fn_qi2.txt"))

print()
print(len(fn1), len(fnqi), len(fnqi2))
print(len(fn1 & fnqi), len(fnqi & fnqi2), len(fn1 & fnqi2))