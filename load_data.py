import numpy as np
import pandas as pd

import datetime
import time

from sklearn.preprocessing import Imputer

def get_data():
    train_data = load_data("Data/train.csv", 0)
    train_data = clean_trips(train_data)
    train_data = clean_date_time(train_data)
    train_data = clean_status(train_data)
    train_data = clean_vessels(train_data)
    train_data.pop('Trip.Duration')

    test_data = load_data("Data/test.csv", 0)
    test_data = clean_trips(test_data)
    test_data = clean_date_time(test_data)
    test_data = clean_vessels(test_data)


    return train_data, test_data

def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

def clean_trips(data, train=True):
    trip = data.pop('Trip')
    data['Tsawwassen to Swartz Bay'] = (trip == 'Tsawwassen to Swartz Bay')*1.0
    data['Tsawwassen to Duke Point'] = (trip == 'Tsawwassen to Duke Point')*1.0
    data['Swartz Bay to Fulford Harbour (Saltspring Is.)'] = (trip == 'Swartz Bay to Fulford Harbour (Saltspring Is.)')*1.0
    data['Swartz Bay to Tsawwassen'] = (trip == 'Swartz Bay to Tsawwassen')*1.0
    data['Duke Point to Tsawwassen'] = (trip == 'Duke Point to Tsawwassen')*1.0
    data['Departure Bay to Horseshoe Bay'] = (trip == 'Departure Bay to Horseshoe Bay')*1.0
    data['Horseshoe Bay to Snug Cove (Bowen Is.)'] = (trip == 'Horseshoe Bay to Snug Cove (Bowen Is.)')*1.0
    data['Horseshoe Bay to Departure Bay'] = (trip == 'Horseshoe Bay to Departure Bay')*1.0
    data['Horseshoe Bay to Langdale'] = (trip == 'Horseshoe Bay to Langdale')*1.0
    data['Langdale to Horseshoe Bay'] = (trip == 'Langdale to Horseshoe Bay')*1.0
    return data

def clean_date_time(data):
    dates = []
    for date in data['Full.Date']:
        dates.append(datetime.datetime.strptime(date, '%d %B %Y').timestamp())

    data.insert(1, 'timestamp', dates)
    data = data.drop(columns=['Month','Day.of.Month','Year','Full.Date'])
    
    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    days = []
    for day in data['Day']:
        days.append(day_names.index(day))

    data = data.drop(columns=['Day'])
    data.insert(1, 'Day', days)
    
    midnight = datetime.datetime.strptime("12:00 AM", "%I:%M %p")
    times = []
    for time in data['Scheduled.Departure']:
        times.append(datetime.datetime.strptime(time, "%I:%M %p").timestamp() - midnight.timestamp())
    
    data = data.drop(columns=['Scheduled.Departure'])
    data.insert(1, 'Scheduled.Departure', times)

    return data

def clean_status(data):
    status_names = dict()
    status_index = 0
    statuses = []
    for status in data['Status']:
        if status not in status_names:
            status_names[status] = status_index
            status_index += 1
        statuses.append(status_names[status])
    
    data = data.drop(columns=['Status'])
    # data.insert(1, 'Status', statuses)
    return data

def clean_vessels(data):
    vessel_name = data.pop('Vessel.Name')
    data['Spirit of British Columbia'] = (vessel_name == 'Spirit of British Columbia')*1.0
    data['Queen of New Westminster'] = (vessel_name == 'Queen of New Westminster')*1.0
    data['Spirit of Vancouver Island'] = (vessel_name == 'Spirit of Vancouver Island')*1.0
    data['Coastal Celebration'] = (vessel_name == 'Coastal Celebration')*1.0
    data['Queen of Alberni'] = (vessel_name == 'Queen of Alberni')*1.0
    data['Coastal Inspiration'] = (vessel_name == 'Coastal Inspiration')*1.0
    data['Skeena Queen'] = (vessel_name == 'Skeena Queen')*1.0
    data['Coastal Renaissance'] = (vessel_name == 'Coastal Renaissance')*1.0
    data['Queen of Oak Bay'] = (vessel_name == 'Queen of Oak Bay')*1.0
    data['Queen of Cowichan'] = (vessel_name == 'Queen of Cowichan')*1.0
    data['Queen of Capilano'] = (vessel_name == 'Queen of Capilano')*1.0
    data['Queen of Surrey'] = (vessel_name == 'Queen of Surrey')*1.0
    data['Queen of Coquitlam'] = (vessel_name == 'Queen of Coquitlam')*1.0
    data['Bowen Queen'] = (vessel_name == 'Bowen Queen')*1.0
    data['Queen of Cumberland'] = (vessel_name == 'Queen of Cumberland')*1.0
    data['Island Sky'] = (vessel_name == 'Island Sky')*1.0
    data['Mayne Queen'] = (vessel_name == 'Mayne Queen')*1.0
    return data