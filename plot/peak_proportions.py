import numpy as np

# Forgive me, Father, for I have sinned.

arr_0 = np.array([67.62192118, 134.75246305, 202.73275862])
arr_1 = np.array([68.26244344, 134.50678733, 202.40723982])
arr_2 = np.array([62.41992188, 122.60742188, 185.30273438])
arr_3 = np.array([62.28301887, 123.41509434, 184.54716981])
arr_4 = np.array([85.2943662, 171.16760563, 255.71971831])
arr_5 = np.array([85.33695652, 156.70652174, 254.29347826])
arr_6 = np.array([56.58256881, 112.20642202, 166.53669725, 223.45412844])
arr_7 = np.array([57.06164384, 113.56849315, 170.07534247, 226.58219178])
arr_8 = np.array([68.84422111, 133.08542714, 199.2160804, 272.90452261])

arr_arr = [arr_0, arr_1, arr_2, arr_3, arr_4, arr_5, arr_6, arr_7, arr_8]

for arr in arr_arr:
    proportion = []
    for val in arr:
        proportion.append(val / arr[0])
    print(proportion)
