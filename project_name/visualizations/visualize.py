# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:02:42 2024

@author: DEPEI_WANG
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.DataFrame([(4.05, 2.52), (2.10, 2.48), (1.90, 2.48), (1.73, 2.52), (1.61, 2.56)])


# print(df)
def visulise(df):
    # Use plot styling from seaborn.
    sns.set(style="darkgrid")

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df[0], "b-o", label="Training")
    plt.plot(df[1], "g-o", label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()


# visulise(df1)
