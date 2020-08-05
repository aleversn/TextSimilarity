# %%
import datetime
import numpy as np

current_time_stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# %%
def WriteSDC(name, info):
    with open("./log/{}.txt".format(name), mode="a+", encoding="utf-8") as f:
        f.write(info)