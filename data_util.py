#!/usr/bin/python

import pandas as pd

class DisplayUtil:
    
    @staticmethod
    def display_all(df):
        with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
            display(df) 