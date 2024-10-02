import numpy as np

linrange = 2*np.pi*np.linspace(1, 2, 100)

for cnt, elem in enumerate(linrange):
    print(f"Element {cnt} is {elem}")
