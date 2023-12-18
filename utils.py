import math

def get_l2dis(a1, a2, b1, b2):
    return math.sqrt((a1 - b1) * (a1 - b1) + (a2 - b2) * (a2 - b2))