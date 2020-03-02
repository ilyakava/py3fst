import math

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))
def prev_power_of_2(x):
    return 1 if x == 0 else 2**(math.ceil(math.log2(x))-1)
def closest_power_of_2(x):
    n = next_power_of_2(x)
    p = prev_power_of_2(x)
    if abs(x-n) < abs(x-p):
        return n
    else:
        return p