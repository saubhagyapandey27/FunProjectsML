import os
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Path to the service account key file
SERVICE_ACCOUNT_FILE = 'path/to/your/credentials.json'

# Scopes required for Google Sheets API
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

# Google Sheet ID and range to read
SPREADSHEET_ID = '1yTedLYGJ8z-X2L7pNPyRnzJpelzzuKH_26vZSBfFjIY'
RANGE_NAME = 'Sheet1!A1:Z1000'  # Adjust range as needed

def main():
    # Authenticate using the service account key
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    # Build the service
    service = build('sheets', 'v4', credentials=credentials)

    # Call the Sheets API to read data
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
    values = result.get('values', [])

    if not values:
        print('No data found.')
    else:
        # Convert the data to a DataFrame and save to a CSV file
        df = pd.DataFrame(values)
        df.to_csv('google_sheet_data.csv', index=False, header=False)
        print('Data saved to google_sheet_data.csv')

if __name__ == '__main__':
    main()
