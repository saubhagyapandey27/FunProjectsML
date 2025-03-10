#Hello, This is Saubhagya Pandey (230939,IITK).
#Before executing this .py file, kindly install the pandas and requests libraries.
#Given API key is generated by me. You can generate your own API key by going to this link: https://console.cloud.google.com/marketplace/product/google/sheets.googleapis.com

import pandas as pd
import requests

# My Google Sheets API key
API_KEY = 'AIzaSyAX8ZkQ3vgEnX19V0WSxnnU0ZTC7KRVhio' 

# Google Sheet ID (extracted from url) and range to read
SPREADSHEET_ID = '1yTedLYGJ8z-X2L7pNPyRnzJpelzzuKH_26vZSBfFjIY'
SHEET_NAMES=['FY 2024-2025','Name change','FY 21-22, 22-23 & 23-24']

# Defining the function
def extract_sheet(SHEET_NAME):
    # URL for the API request
    RANGE_NAME = f'{SHEET_NAME}!A1:Z2500' 
    url = f'https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values/{RANGE_NAME}?key={API_KEY}'
    
    # Making a Request and Getting a response
    response = requests.get(url)
    if response.status_code != 200:
        print('Failed to get data')
        return
    
    data = response.json()
    values = data.get('values')
    
    if not values:
        print('No data found on the Sheet')
    else:
        # Converting the data to a DataFrame and saving to a CSV file
        df = pd.DataFrame(values)
        df.to_csv(f'{SHEET_NAME}.csv', index=False, header=False)
        print(f"{SHEET_NAME} Sheet's Data saved to CSV File")

# Extracting the data
for i in SHEET_NAMES:
    extract_sheet(i)

#Thank You