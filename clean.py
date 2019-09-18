import numpy as np
import pandas as pd

import datetime
import time

# test data:
# "ID","Vessel.Name","Scheduled.Departure","Trip","Day","Month","Day.of.Month","Year","Full.Date"

# train data: 
# "Vessel.Name","Scheduled.Departure","Status","Trip","Trip.Duration","Day","Month","Day.of.Month","Year","Full.Date","Delay.Indicator"

def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

def clean_trips(data, train=True):
    places = dict()
    places_index = 0
    froms = []
    dests = []
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
    
    data.insert(1, 'beginning', froms)
    data.insert(1, 'destination', dests)
    data = data.drop(columns=['Trip'])
    if train:
        data = data.drop(columns=['Trip.Duration'])
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
    vessel_names = dict()
    vessel_index = 0
    vessels = []
    for vessel in data['Vessel.Name']:
        if vessel not in vessel_names:
            vessel_names[vessel] = vessel_index
            vessel_index += 1
        vessels.append(vessel_names[vessel])

    data = data.drop(columns=['Vessel.Name'])
    data.insert(1, 'Vessel', vessels)
    return data
