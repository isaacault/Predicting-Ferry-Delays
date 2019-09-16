import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import datetime
import time

def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

def clean_trips(data):
    places = dict()
    places_index = 0
    for trip in data['Trip']:
        locs = trip.split(" to ")
        beg = locs[0]
        dest = locs[1]
        if beg not in places:
            places[beg] = places_index
            places_index += 1
        if dest not in places:
            places[dest] = places_index
            places_index += 1

        froms.append(places[beg])
        dests.append(places[dest])
    
    data.insert(3, 'beginning', froms)
    data.insert(4, 'destination', dests)
    data = data.drop(columns=['Trip'])
    print(data.columns)
    return data

def clean_date_time(data):
    dates = []
    for date in data['Full.Date']:
        dates.append(datetime.datetime.strptime(date, '%d %B %Y').timestamp())

    data.insert(6, 'timestamp', dates)
    data = data.drop(columns=['Month','Day.of.Month','Year','Full.Date'])
    
    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    days = []
    for day in data['Day']:
        days.append(day_names.index(day))

    data = data.drop(columns=['Day'])
    data.insert(5, 'Day', days)
    
    midnight = datetime.datetime.strptime("12:00 AM", "%I:%M %p")
    times = []
    for time in data['Scheduled.Departure']:
        times.append(datetime.datetime.strptime(time, "%I:%M %p").timestamp() - midnight.timestamp())
    
    data = data.drop(columns=['Scheduled.Departure'])
    data.insert(4, 'Scheduled.Departure', times)

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
    data.insert(1, 'Status', statuses)
    return data

def clean_vessels(data):
    vessel_names = dict()
    vessel_index = 0
    vessels = []
    for vessel in data['Vessel.Name']:
        if vessel not in vessel_names:
            vessel_names[vessel] = vessel_index
            vessel_index += 1
        vessels.append(vessel_names[vessel])

    data = data.drop(columns=['Vessel.Name'])
    data.insert(0, 'Vessel', vessels)
    return data

if __name__ == "__main__":
    data = load_data("Data/train.csv", 0)
    
    froms = []
    dests = []
   

    data = clean_trips(data)
    data = clean_date_time(data) 
    data = clean_status(data)
    data = clean_vessels(data)

    X = data[['beginning', 'destination']]

    # y = target values, last column of the data frame
    y = data['Delay.Indicator']

    # filter out the applicants that got admitted
    delayed = data.loc[y == 1]

    # filter out the applicants that din't get admission
    not_delayed = data.loc[y == 0]

    # plots
    plt.scatter(delayed['beginning'], delayed['destination'], s=10, label='Delayed')
    plt.scatter(not_delayed['beginning'], not_delayed['destination'], s=10, label='Not Delayed')
    plt.legend()
    plt.savefig('/mnt/c/Users/rkane/Pictures/plot.png')
