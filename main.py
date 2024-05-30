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

    Used as a helper in process_wage_growth()
    """
    skiprows = 1 if sheet == 'data_overall' else 2
    excel_output = pd.read_excel('wage_growth_data.xlsx',
                                 sheet_name=sheet, skiprows=skiprows)

    return excel_output

def process_wage_growth() -> pd.DataFrame:
    """
    Reads in url from Fed. Reserve in Atlanta, creates a dataframe
    based off selected sheets, and saves the sheets chosen as
    a new csv. Returns the pandas Dataframe object of the .csv.
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
    Reads in url from Fed. Reserve St. Louis, renames columns for clarity,
    and saves the columns as a csv file. Returns the pandas Dataframe object of the csv.
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

def process_minimum_wage() -> pd.DataFrame:
    """
    Reads in MinWage_PartyControl.csv and MinimumWage.csv and creates new
    merged_df that combines important columns selected. Creates dates dataframe
    and merges to filtered_df to create a dates column by month. Assigns dates
    column as the first column, and creates a year column for later reference.
    Saves filtered_df as a csv file. Returns the pandas Dataframe object of the .csv.
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

def process_interest_rate() -> pd.DataFrame:
    """
    Reads in url from Fed. Reserve of St. Louis and reads it in, selecting
    the column 3MonthInterestRate to be created as a new csv. Returns the 
    pandas Dataframe object of the .csv.
    """
    url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0'\
        '&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&'\
        'height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&'\
        'tts=12&width=1138&nt=0&thu=0&trc=0&show_legend=yes&'\
        'show_axis_titles=yes&show_tooltip=yes&id=IR3TIB01USM156N&scale=left'\
        '&cosd=1964-06-01&coed=2024-04-01&line_color=%234572a7'\
        '&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2'\
        '&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin'\
        '&fgsnd=2020-02-01&line_index=1&transformation=lin'\
        '&vintage_date=2024-05-20&revision_date=2024-05-20&nd=1964-06-01'
    df = pd.read_csv(url)

    #Rename the second column to be more clear
    df.columns.values[1] = '3MonthInterestRate'
    df.to_csv('3-Month Interest Rates', index=False)

    return df

def process_cpi() -> pd.DataFrame:
    """
    Reads in cpi.csv from usinflationcalculator.com, which uses the CPI calculator function
    from the US Department of Labor. Cleans and drops unncessary dates to keep in line with date
    ranges in dataset. Pivots wide by utilizing melt function from pandas, and maps each month as
    its own row. Unmelts the pivoted dataframe, filters dates to only include those relevant to the
    dataset, and saves the pivoted dataframe as new_cpi.csv. Then normalizes CPI to be month over 
    month, as this is in line with the observations in the dataset being wage change by month. The 
    newly created column is then saved as CPI_Month_Over_Month_Changes.csv. Returns the pandas
    Dataframe object as a .csv.
    """
    # read in cpi csv
    cpi = pd.read_csv('cpi.csv')

    # omit first column
    cpi = cpi.iloc[:, 1:]

    # allows us to drop converting string to int type
    cpi['Year'] = cpi['Year'].astype(int)

    # sets year as index
    cpi = cpi.set_index("Year")

    # drops all unnecessary years and columns
    cpi = cpi.drop(index = range(1913, 1996))
    cpi = cpi.drop(columns = ["Avg", "Dec-Dec", "Avg-Avg"])
    # 27 rows by 15 columns

    # use melt function from pandas to transform cpi df
    cpi_melted = cpi.reset_index().melt(id_vars='Year',
                                        var_name='Month', value_name='CPI')
    month_designation = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                         'May': '05', 'June': '06', 'July': '07', 'Aug': '08',
                         'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    cpi_melted['Month'] = cpi_melted['Month'].map(month_designation)
    cpi_melted['Date'] = pd.to_datetime(cpi_melted['Year'].astype(str) \
                                        + '-' + cpi_melted['Month'] + '-01')
    cpi_melted = cpi_melted.drop(columns=['Year', 'Month'])
    cpi_melted = cpi_melted.set_index('Date')
    cpi_melted = cpi_melted.sort_values('Date')

    # only include dates applicable to original df
    new_cpi = cpi_melted[cpi_melted.index < '2024-04-01']
    new_cpi.to_csv('new_cpi.csv')

    # Normalize CPI data above to be month over month, then creating a new 
    # dataframe with only date and month over month calculations.

    # Calculate the month-over-month percentage change
    new_cpi['MoM Change'] = new_cpi['CPI'].pct_change() * 100
    selected_columns = new_cpi.reset_index()[['Date', 'MoM Change']]

    # Save the resulting DataFrame to a new CSV file
    selected_columns.to_csv('CPI_Month_Over_Month_Changes.csv', index=False)

    return selected_columns

def create_final_dataframe() -> pd.DataFrame:
    """
    Reads in previously created .csvs and filters by desirable dates. 
    The .csvs are checked for its datetime datatypes, and is merged into the
    finalData dataframe, which is then saved for redundancy. Drops unnecessary columns, 
    merges with previously created .csvs and is saved as a .csv. Returns a pandas
    Dataframe object of the .csv.
    """
    # Load and filter the large DataFrame
    merged_df = pd.read_csv('wageGrowth.csv', parse_dates=[0])
    filtered_df = merged_df[merged_df.iloc[:, 0] > '1997-12-01']
    # Rename the first column
    filtered_df.columns = ['Date'] + list(filtered_df.columns[1:])

    # Load and filter the 3-month interest rates DataFrame
    rates = pd.read_csv('3-Month Interest Rates', parse_dates=[0])
    filtered_rates = rates[rates.iloc[:, 0] > '1997-12-01']
    filtered_rates.columns = ['Date'] \
        + list(filtered_rates.columns[1:])  # Rename the first column

    # Ensure that both 'Date' columns are in the datetime64[ns] format
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
    filtered_rates['Date'] = pd.to_datetime(filtered_rates['Date'])

    # Merge the two filtered DataFrames on the "Date" column
    finalData = pd.merge(filtered_df, filtered_rates, on='Date')

    # Save the merged DataFrame to a new CSV file
    finalData.to_csv('merged_filtered_data.csv', index=False)

    # Drop any 'Unnamed' columns that might be there due to indexing
    for i in range(13):
        finalData = finalData.drop(columns=f"Unnamed: 0.{i}", errors='ignore')

    # Read in the new CPI data and merge
    newcpi = pd.read_csv('CPI_Month_Over_Month_Changes.csv')
    finalData = finalData.merge(newcpi, how='right', on='Date')

    # Read in the employment rate data and merge
    emplyRate = pd.read_csv('Employment_Rate.csv')
    finalData = finalData.merge(emplyRate, how='right', on='Date')

    # Read in the minimum wage data and merge
    minimumWage = pd.read_csv('mergedMinimumWageData.csv')
    finalData = finalData.merge(minimumWage, how='left', on='Date')

    # Save the final merged DataFrame to a new CSV file
    finalData.to_csv('finalData.csv', index=False)

    return finalData

def OLS_1() -> object:
    """
    Regressing Overall.12 (overall % change in wage growth, monthly, for the whole dataset) 
    over 3MonthInterestRate, MoM Change, FedMinWage, and GDP_AnnualGrowth
    Reads in our final dataset and cleans/our data in order to run OLS.
    Utilizes statsmodel OLS function which runs OLS and returns model summary.
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

    # assign dummy variables for democrat (0) and republican (1)
    def map_party(party):
        if party == 'Democrat':
            return 0
        elif party == 'Republican':
            return 1
        else:
            return np.nan

    # Apply the mapping function to the party columns
    data['PresParty'] = data['PresParty'].apply(map_party)
    data['SenParty'] = data['SenParty'].apply(map_party)
    data['HouseParty'] = data['HouseParty'].apply(map_party)

    # Drop rows with NA values
    data = data.dropna(subset=['PresParty', 'SenParty', 'HouseParty','Overall.12','3MonthInterestRate', 'MoM Change', 'Employment_Rate', 'FedMinWage', 'GDP_AnnualGrowth'])

    # Separate the dependent variable (Y) and independent variables (X)
    Y_filtered = data['Overall.12']
    X_filtered = data[['PresParty', 'SenParty', 'HouseParty', '3MonthInterestRate', 'MoM Change', 'FedMinWage', 'GDP_AnnualGrowth']]

    # Add a constant term to the predictors
    X_filtered = sm.add_constant(X_filtered)

    # Fit the OLS regression model
    ols_model = sm.OLS(Y_filtered, X_filtered).fit()

    # Print the summary of the regression results
    sum_OLS = ols_model.summary()
    
    return sum_OLS

def OLS_2() -> object:

    """
    Regressed additional variables like Male, Female, Bachelors degree or higher, and
    others on Overall.12. Reads in our final dataset and cleans/our data in order to run OLS.
    Utilizes statsmodel OLS function which runs OLS and returns model summary.
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

    Y = data['Overall.12'].dropna()
    Y = Y[(data['Date'] >= '1998-01-01') & (data['Date'] <= '2020-12-01')]
    # Define the independent variables, add a constant term to the predictors
    X = data[['3MonthInterestRate', 'MoM Change', 'FedMinWage', 'GDP_AnnualGrowth', 'Job Stayer', 
              'Job Switcher', 'Male', 'Female', 'Bachelors degree or higher']]
    X = X.dropna(subset=['3MonthInterestRate', 'MoM Change',  'FedMinWage', 'GDP_AnnualGrowth', 
                         'Job Stayer', 'Job Switcher', 'Male', 'Female', 'Bachelors degree or higher'])
    X = sm.add_constant(X)

    # Create the OLS model
    est = sm.OLS(Y, X.astype(float)).fit()

    sum_OLS_2 = est.summary()
    return sum_OLS_2

def corr_matrices_and_visuals() -> object:
    """
    Chooses variables from final dataset and drops nas for use in correlation matrices.
    Utilizes .corr() function from pandas library. Utilizes heatmap function from seaborn
    and returns correlation heatmaps with different palettes alongside correlation matrices.
    """

    # read in the dataset
    data = pd.read_csv('C:\\Users\\danny\\481\\481\\Econ-481-Final-Project\\finalData.csv')

    # modifies data columns to be used as part of correlation matrix
    data['3MonthInterestRate'] = pd.to_numeric(data['3MonthInterestRate'], errors='coerce')
    data['FedMinWage'] = data['FedMinWage'].str.replace('$', '').astype(float)
    data['GDP_AnnualGrowth'] = data['GDP_AnnualGrowth'].str.replace('%', '').astype(float) / 100

    # choose the following variables for correlation matrix using dataframe X
    X = data[['3MonthInterestRate', 'MoM Change', 'Employment_Rate', 'FedMinWage', 'GDP_AnnualGrowth']]
    X = X.dropna()

    # choose the following variables for dataframe X_2
    X_2 = data[['FedMinWage', 'MoM Change', '3MonthInterestRate', 'GDP_AnnualGrowth']]
    X_2 = X_2.dropna()

    # choose the following variables for dataframe X_3
    X_3 = data[['3MonthInterestRate', 'MoM Change', 'Employment_Rate', 'FedMinWage',
                       'GDP_AnnualGrowth', 'Lowest quartile of wage distribution',
                       '2nd quartile of wage distribution', '3rd quartile of wage distribution',
                       'Highest quartile of wage distribution']]
    X_3 = X_3.dropna()

    # assign each correlation matrix as its own variable
    correlation_matrix_1 = X.corr()
    correlation_matrix_2 = X_2.corr()
    correlation_matrix_3 = X_3.corr()

    # output formatting
    print("Variance Inflation Factors for X:")
    print(correlation_matrix_1)
    print("\nVariance Inflation Factors for X_2:")
    print(correlation_matrix_2)
    print("\nVariance Inflation Factors for X_3:")
    print(correlation_matrix_3)
    
    # plot and visualize heatmaps
    plt.figure(figsize=(10, 8))
    map_1 = sns.heatmap(correlation_matrix_1, annot=True, cmap=sns.diverging_palette(50, 500, n = 500), vmin=-1, vmax=1)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    plt.figure(figsize=(10, 8))
    map_2= sns.heatmap(correlation_matrix_2, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    plt.figure(figsize=(10, 8))
    map_3 = sns.heatmap(correlation_matrix_3, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    # save each heat map as apng
    plt.savefig(map_1)
    plt.savefig(map_2) 
    plt.savefig(map_3)

    return correlation_matrix_1, correlation_matrix_2, correlation_matrix_3, map_1, map_2, map_3

print(corr_matrices_and_visuals())

def vif_matrix() -> object:
    """
    Reads in dataset, and drops nas for each variable in order to be used in variance 
    inflation factor calculations. Returns  VIF matrices for each X, X_2, X_3
    """
    # reads in data
    data = pd.read_csv('finalData.csv')

    # modifies data columns to be used as part of correlation matrix
    data['3MonthInterestRate'] = pd.to_numeric(data['3MonthInterestRate'], errors='coerce')
    data['FedMinWage'] = data['FedMinWage'].str.replace('$', '').astype(float)
    data['GDP_AnnualGrowth'] = data['GDP_AnnualGrowth'].str.replace('%', '').astype(float) / 100

    # choose the following variables for dataframe X
    X = data[['3MonthInterestRate', 'MoM Change', 'Employment_Rate', 'FedMinWage', 'GDP_AnnualGrowth']]
    X = X.dropna()

    # choose the following variables for dataframe X_2
    X_2 = data[['FedMinWage', 'MoM Change', '3MonthInterestRate', 'GDP_AnnualGrowth']]
    X_2 = X_2.dropna()

    # choose the following variables for dataframe X_3
    X_3 = data[['3MonthInterestRate', 'MoM Change', 'Employment_Rate', 'FedMinWage',
                       'GDP_AnnualGrowth', 'Lowest quartile of wage distribution',
                       '2nd quartile of wage distribution', '3rd quartile of wage distribution',
                       'Highest quartile of wage distribution']]
    X_3 = X_3.dropna()

    # creates a vif object where each element is its on vif value based on feature type
    vif_1 = pd.DataFrame()
    vif_1["Feature"] = X.columns
    vif_1["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    vif_2 = pd.DataFrame()
    vif_2["Feature"] = X_2.columns
    vif_2["VIF"] = [variance_inflation_factor(X_2.values, i) for i in range(len(X_2.columns))]

    vif_3 = pd.DataFrame()
    vif_3["Feature"] = X_3.columns
    vif_3["VIF"] = [variance_inflation_factor(X_3.values, i) for i in range(len(X_3.columns))]

    # output formatting
    print("Variance Inflation Factors for X:")
    print(vif_1)
    print("\nVariance Inflation Factors for X_2:")
    print(vif_2)
    print("\nVariance Inflation Factors for X_3:")
    print(vif_3)

    return vif_1, vif_2, vif_3

def main():
    """
    Calls our functions to wrangle and save data, create graphs, and run 
    analyses.
    """
    process_wage_growth()
    process_employment_rate()
    process_minimum_wage()
    process_interest_rate()
    process_cpi()
    finalData = create_final_dataframe()
    OLS_1()
    OLS_2()
    corr_matrices_and_visuals()
    vif_matrix()

if __name__ == '__main__':
    main()