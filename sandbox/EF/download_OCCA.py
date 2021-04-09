import os
import urllib.request


PATH_OCCA = os.path.expanduser('~/work/data/OCCA/')
os.makedirs(PATH_OCCA, exist_ok=True)

varnames = "theta salt phihyd etan".split()
ftp = "ftp://mit.ecco-group.org/ecco_for_las/OCCA_1x1_v2/2004-6/annual/"

for name in varnames:
    fname = f"DD{name}.0406annclim.nc"
    url = f"{ftp}{fname}"
    dest = f"{PATH_OCCA}{fname}"
    print("retrieving:", url, dest)
    ret = urllib.request.urlretrieve(url, dest)
    print("returned:", ret)
