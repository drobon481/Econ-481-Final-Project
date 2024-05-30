"""
Zach Koverman, Henry Tran, David Robon
Lukas Hager
ECON 481
29 May 2024


Final Project - All Code
Analysis of Wage Inflation Trends and Correlations

For our final project, we looked into wage growth in the United States and 
analyzed how it breaks down and the relationships between the factors that 
contribute to it. This file contains all the code we used to clean, merge, and 
manipulate our data; graphically explore patterns in the relationships between 
certain variables; examine any multicollinearity between variables; and run OLS 
regression to investigate how much certain variables contribute to wage growth.


Our data sources:

Federal Reserve Bank of Atlanta - Wage Growth
https://www.atlantafed.org/chcs/wage-growth-tracker
- 1998-2024 monthly percent change in wage in the United States
- Contains columns for overall change as well as how wages changed for 
  different worker demographics

Federal Reserve Bank of St. Louis - 

"""

import requests
import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt


def read_sheet(sheet) -> pd.DataFrame:
    """
    This function reads given excel sheets using different skiprows parameters.
    It skips the first row for data_overall tab and skips the first two rows on
    all other tabs. Returns the pandas DataFrame object of the read .xlsx file.
    """
    skiprows = 1 if sheet == 'data_overall' else 2
    excel_output = pd.read_excel('wage_growth_data.xlsx',
                                 sheet_name=sheet, skiprows=skiprows)

    return excel_output

def process_wage_growth() -> pd.DataFrame:
    """
    
    """
    # Download and save the Excel file
    url = 'https://www.atlantafed.org/-/media/documents/datafiles/chcs/'\
          'wage-growth-tracker/wage-growth-data.xlsx'
    with open('wage_growth_data.xlsx', 'wb') as file:
        file.write(requests.get(url).content)

    # List of sheet names on Wage Growth excel sheet
    sheet_names = [
        'Education', 'Age', 'Sex', 'Occupation', 'Industry', 'Census Divisions',
        'Full-Time or Part-Time', 'Job Switcher', 'MSA or non-MSA', 
        'Average Wage Quartile', 'Paid Hourly', 'Overall 12ma', 'data_overall'
    ]

    # Merge sheets
    merged_df = pd.concat([read_sheet(sheet) for sheet in sheet_names], axis=1)

    # Save to a CSV file
    merged_df.to_csv('wageGrowth.csv', index=False)

    return merged_df

def process_employment_rate() -> pd.DataFrame:
    """
    
    """
    url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0'\
        '&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff'\
        '&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12'\
        '&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes'\
        '&show_axis_titles=yes&show_tooltip=yes&id=LREM64TTUSM156S&scale=left'\
        '&cosd=1977-01-01&coed=2024-04-01&line_color=%234572a7&'\
        'link_values=false&line_style=solid&mark_type=none&mw=3&lw=2'\
        '&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin'\
        '&fgsnd=2020-02-01&line_index=1&transformation=lin'\
        '&vintage_date=2024-05-22&revision_date=2024-05-22&nd=1977-01-01'

    df = pd.read_csv(url)

    #Rename the second column to be more clear
    df = df.rename(columns={df.columns[1]: 'Employment_Rate'})
    df = df.rename(columns={df.columns[0]: 'Date'})

    df.to_csv('Employment_Rate.csv', index=False)

    return df

def process_minimumwage() -> pd.DataFrame:
    """
    
    """
    # Load data
    minimumWage_Party = pd.read_csv('MinWage_PartyControl.csv').iloc[:, :6]
    minimumWage = pd.read_csv('MinimumWage.csv')
    # Assuming 'Year' is already a column, not needing reset_index()
    selected_columns = minimumWage[['Year', 'GDP_AnnualGrowth']]

    # Merge the datasets
    merged_df = pd.merge(minimumWage_Party, selected_columns,
                         on='Year', how='inner')
    filtered_df = merged_df[merged_df['Year'] > 1976]

    # Generate a DataFrame with all months for each year
    dates = pd.date_range(start=f"{filtered_df['Year'].min()}-01-01",
                          end=f"{filtered_df['Year'].max()}-12-31", freq='MS')
    dates_df = pd.DataFrame({'Date': dates})

    # Merge the dates with the filtered DataFrame based on the year
    filtered_df = filtered_df.merge(dates_df, left_on='Year',
                                    right_on=dates_df['Date'].dt.year)

    # Reorder to make 'Date' the first column and rename it
    filtered_df.drop(columns=['Year'], inplace=True)
    # Recreate 'Year' if needed elsewhere
    filtered_df['Year'] = filtered_df['Date'].dt.year  
    filtered_df = filtered_df\
        [['Date'] + [col for col in filtered_df.columns if col != 'Date']]

    # Save the final DataFrame
    filtered_df.to_csv('mergedMinimumWageData.csv', index=False)

    return filtered_df


def OLS_1() -> object:
    """
    Regressing Overall.12 (overall % change in wage growth, monthly, for the whole dataset) 
    over 3MonthInterestRate, MoM Change, FedMinWage, and GDP_AnnualGrowth
    Reads in our final dataset and cleans/our data in order to run OLS.
    Utilizes statsmodel OLS function which runs OLS and outputs model summary
    """
    data = pd.read_csv('finalData.csv')

    # Convert the '3MonthInterestRate' column to string type
    data['3MonthInterestRate'] = data['3MonthInterestRate'].astype(str)

    # Remove decimal points only when there is no number before or after it
    data['3MonthInterestRate'] = data['3MonthInterestRate'].str.replace(r'(?<!\d)\.|\.(?!\d)', '', regex=True)

    # Replace blanks with '0'
    data['3MonthInterestRate'] = data['3MonthInterestRate'].replace('', '0')

    # Convert the column to float
    data['3MonthInterestRate'] = data['3MonthInterestRate'].astype(float)

    # Correctly remove dollar signs from 'FedMinWage' and convert to float
    data['FedMinWage'] = data['FedMinWage'].str.replace('$', '').astype(float)

    # Correctly remove percent signs from 'GDP_AnnualGrowth', convert to float, and adjust for percentage representation
    data['GDP_AnnualGrowth'] = data['GDP_AnnualGrowth'].str.replace('%', '').astype(float) / 100

    # Display the corrected data types to ensure the operations were successful
    print(data[['FedMinWage', 'GDP_AnnualGrowth', '3MonthInterestRate']].dtypes)
    
    # Filter Y variables
    Y = data['Overall.12']
    Y_filtered = Y[(data['Date'] >= '1998-01-01') & (data['Date'] <= '2020-12-01')]

    # Filter X variables
    X = data[['3MonthInterestRate', 'FedMinWage', 'GDP_AnnualGrowth']]
    X_filtered = X[(data['Date'] >= '1998-01-01') & (data['Date'] <= '2020-12-01')]
    X = sm.add_constant(X)
    # Reset indices to align
    Y_filtered = Y_filtered.reset_index(drop=True)
    X_filtered = X_filtered.reset_index(drop=True)

    # Add constant to X
    X_filtered = sm.add_constant(X_filtered)

    # Create the OLS model
    est = sm.OLS(Y_filtered, X_filtered.astype(float)).fit()
    sum_OLS_1 = est.summary()
    

    return sum_OLS_1

print(OLS_1())


def main():
    """
    Calls our functions to wrangle data, create graphs, and run analysis.
    """
    merged_df = process_wage_growth()


if __name__ == '__main__':
    main()