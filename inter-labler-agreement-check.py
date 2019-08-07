#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 17:00:23 2018

@author: fraifeld-mba
"""

import pandas as pd
def load_labeled_reviews(filename):
    df = pd.read_json(filename, lines = True)
    return df


def calculate_ila(df1, df2):
    merged = df1.merge(df2, on="review_id")
    
    print("Disagreement on: ")
    disagreement = merged[merged.label_x != merged.label_y].reset_index(drop=True)
    for i in range(5,7):
        print(disagreement["text_x"].values[i])
        print("Abe: " + str(disagreement.loc[i,"label_x"]))
        print("Youhui: " +str(disagreement.loc[i,"label_y"]))

        print("\n")

    return len(merged[merged.label_x == merged.label_y])/len(merged)

def re_label(df, category):
    if category == "food":
        df.loc[df.label=="f","label"] = 1
        df.loc[df.label=="b","label"] = 1
        df.loc[df.label!=1,"label"] = 0


    else:
        df.loc[df.label=="s","label"] = 1
        df.loc[df.label=="b","label"] = 1
        df.loc[df.label!=1,"label"] = 0

    return df

if __name__ == "__main__":
    
    youhui = load_labeled_reviews("15000-15100.json")
    abraham = load_labeled_reviews("labeled_reviews.json")

    food_abraham = re_label(abraham,"food")
    abraham = load_labeled_reviews("labeled_reviews.json")
    service_abraham = re_label(abraham, "service")
    
    
    food_youhui = re_label(youhui,"food")
    youhui = load_labeled_reviews("15000-15100.json")
    service_youhui = re_label(youhui, "service")
    
    print("Inter-Labler-Agreement Food: " + str(calculate_ila(food_abraham, food_youhui)))
    print("Inter-Labler-Agreement Service: " + str(calculate_ila(service_abraham, service_youhui)))


