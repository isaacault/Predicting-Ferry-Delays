import numpy as np
import pandas as pd

import datetime
import time

from sklearn.preprocessing import Imputer

# test data:
# "ID","Vessel.Name","Scheduled.Departure","Trip","Day","Month","Day.of.Month","Year","Full.Date"

# train data: 
# "Vessel.Name","Scheduled.Departure","Status","Trip","Trip.Duration","Day","Month","Day.of.Month","Year","Full.Date","Delay.Indicator"

def load_data(path, header=0):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

def clean_trips(data, train=True):
    # places = dict()
    # places_index = 0
    # froms = []
    # dests = []
    # for trip in data['Trip']:
    #     locs = trip.split(" to ")
    #     beg = locs[0]
    #     dest = locs[1]
    #     if beg not in places:
    #         places[beg] = places_index
    #         places_index += 1
    #     if dest not in places:
    #         places[dest] = places_index
    #         places_index += 1
    #     froms.append(places[beg])
    #     dests.append(places[dest])
    
    # data.insert(1, 'beginning', froms)
    # data.insert(1, 'destination', dests)
    # data = data.drop(columns=['Trip'])
    # if train:
    #     data = data.drop(columns=['Trip.Duration'])

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
    # vessel_names = dict()
    # vessel_index = 0
    # vessels = []
    # for vessel in data['Vessel.Name']:
    #     if vessel not in vessel_names:
    #         vessel_names[vessel] = vessel_index
    #         vessel_index += 1
    #     vessels.append(vessel_names[vessel])
    # data = data.drop(columns=['Vessel.Name'])
    # data.insert(1, 'Vessel', vessels)
    # return data
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

def merge_traffic(data, traffic_data):
    # "Year","Month","Day","Hour","Minute","Second","Traffic.Ordinal"
    # make the traffic_data just time and traffic ordinal so that it 
    # can easily merge with our dataset. In the end it will be:
    # "minute_timestamp", "Traffic"
    minute_timestamps = []
    print("iter traffic")
    for index, row in traffic_data.iterrows():
        try:
            ts = datetime.datetime(year=int(row['Year']),
                                month=int(row['Month']),
                                day=int(row['Day']),
                                hour=int(row['Hour']),
                                minute=int(row['Minute']))
        except ValueError as e:
            continue
        minute_timestamps.append(ts.timestamp())
    traffic_data = traffic_data.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])
    traffic_data.insert(1, 'minute_timestamp', minute_timestamps)
    
    minute_timestamps = []
    print("iter train")
    for index, row in data.iterrows():
        ts = datetime.datetime.fromtimestamp(row['timestamp'] + row['Scheduled.Departure'])
        minute_timestamps.append(ts.timestamp - ts.second)
    data.insert(1, 'minute_timestamp', minute_timestamps)
    print("merge")
    print(traffic_data)
    print(data)
    data = pd.merge(data, traffic_data, how='outer')
    print(data)

def stitch_traffic(data, traffic_data):
    # "Year","Month","Day","Hour","Minute","Second","Traffic.Ordinal"
    # for each entry in the data, add a column with the corresponding traffic ordinal
    traffic = []
    for index, row in data.iterrows():
        date = row['timestamp'] + row['Scheduled.Departure']
        date = datetime.datetime.fromtimestamp(date)
        traffic_point = traffic_data.loc[
            (traffic_data['Year'] == date.year) &
            (traffic_data['Month'] == date.month) &
            (traffic_data['Day'] == date.day) &
            (traffic_data['Hour'] == date.hour) &
            (traffic_data['Minute'] == date.minute)
            ]
        if not traffic_point.shape[0]:
            traffic_point = traffic_data.loc[
                    (traffic_data['Month'] == date.month) &
                    (traffic_data['Day'] == date.day) &
                    (traffic_data['Hour'] == date.hour)
                ]
            if not traffic_point.shape[0]:
                traffic_point = traffic_data.loc[
                        (traffic_data['Month'] == date.month)
                    ]['Traffic.Ordinal'].mean()
            else:
                traffic_point = traffic_point.iloc[0]['Traffic.Ordinal']
        else:
            traffic_point = traffic_point.iloc[0]['Traffic.Ordinal']
        progress_perc = int(100*len(traffic)/data.shape[0])
        print("stitching traffic data: " + str(progress_perc) + "% [" + "="*progress_perc + " "*(100-progress_perc) +"] ", end="    \r")
        traffic.append(traffic_point)
    print("\nDone.")
    data.insert(1, 'Traffic', traffic)
    return data

def stitch_weather(data, weather_data, city):
    # vancouver
    # "Date.Time","Year","Month","Day","Time","Temperature.in.Celsius","Dew.Point.Temperature.in.Celsius",
    # "Relative.Humidity.in.Percent","Humidex.in.Celsius","Hour"
    
    # for each entry in the data, add a column with the corresponding weather
    tempsinc = []
    dewpointtempsinc = []
    relhumidinpercents = []
    humidinc = []
    winddirs = []
    windspds = []
    visinkms = []
    stprssrs = []
    for index, row in data.iterrows():
        date = row['timestamp']
        # 2016-08-01 00:00:00
        # date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        date = datetime.datetime.fromtimestamp(date)
        weather_point = weather_data.loc[
            weather_data['Date.Time'] == date.strftime("%Y-%m-%d %H:%M:00")
        ]
        weather_point_similar = weather_data.loc[
            (weather_data['Month'] == str(date.month))
            # (weather_data['Time'] == date.strftime("%H:00"))
        ]
        tempinc = weather_point.iloc[0]['Temperature.in.Celsius']
        dewpointtempinc = weather_point.iloc[0]['Dew.Point.Temperature.in.Celsius']
        relhumidinpercent = weather_point.iloc[0]['Relative.Humidity.in.Percent']
        if str(tempinc) == "nan":
            tempinc = weather_point_similar['Temperature.in.Celsius'].mean()
        if str(dewpointtempinc) == "nan":
            dewpointtempinc = weather_point_similar['Dew.Point.Temperature.in.Celsius'].mean()
        if str(relhumidinpercent) == "nan":
            relhumidinpercent = weather_point_similar['Relative.Humidity.in.Percent'].mean()
        tempsinc.append(tempinc)
        dewpointtempsinc.append(dewpointtempinc)
        relhumidinpercents.append(relhumidinpercent)
        if city == "vancouver":
            huminc = weather_point.iloc[0]['Humidex.in.Celsius']
            if str(huminc) == "nan":
                huminc = weather_point_similar['Humidex.in.Celsius'].mean() 
            humidinc.append(huminc)
        if city == "victoria":
            # victoria
            # "Wind.Direction.in.Degrees","Wind.Speed.km.per.h","Visibility.in.km","Station.Pressure.in.kPa","Weather"
            
            winddir = weather_point.iloc[0]['Wind.Direction.in.Degrees']
            windspd = weather_point.iloc[0]['Wind.Speed.km.per.h']
            visinkm = weather_point.iloc[0]['Visibility.in.km']
            stprssr = weather_point.iloc[0]['Station.Pressure.in.kPa']
            
            if str(winddir) == "nan":
                winddir = weather_point_similar['Wind.Direction.in.Degrees'].mean() 
            winddirs.append(winddir)
            if str(windspd) == "nan":
                windspd = weather_point_similar['Wind.Speed.km.per.h'].mean() 
            windspds.append(winddir)
            if str(visinkm) == "nan":
                visinkm = weather_point_similar['Visibility.in.km'].mean() 
            visinkms.append(winddir)
            if str(stprssr) == "nan":
                stprssr = weather_point_similar['Station.Pressure.in.kPa'].mean() 
            stprssrs.append(winddir)

        progress_perc = int(100*len(tempsinc)/data.shape[0])
        print("stitching " + city + " weather data: " + str(progress_perc) + "% [" + "="*progress_perc + " "*(100-progress_perc) +"] ", end="    \r")
        
    data.insert(1, city + ".TempinC", tempsinc)
    data.insert(1, city + ".DewPointTempInC", dewpointtempsinc)
    data.insert(1, city + ".RelHumidInPercent", relhumidinpercents)
    if city =="vancouver":
        data.insert(1, city + ".HumidInC", humidinc)
    if city =="victoria":
        data.insert(1, city + ".WindDir", winddirs)
        data.insert(1, city + ".WindSpeed", windspds)
        data.insert(1, city + ".VisInKm", visinkms)
        data.insert(1, city + ".StationPressurekPa", stprssrs)
    print("\nDone.")
    return data

def stitch_weather_types(data, weather_data):
    weather_types = weather_data['Weather'].unique().tolist()
    weather = weather_data.pop('Weather')

    for w_type in weather_types:
        print(str(w_type))
        w_type = str(w_type)
        if len(w_type.split(",")) > 1:
            for sub_type in w_type.split(","):
                weather_types.append(sub_type)
            weather_types.remove(w_type)

    for w_type in weather_types:
        w_type = str(w_type)
        data[w_type] = (weather == w_type)*1.0
    return data

def clean_traffic_data(data):
    data.dropna()
    timestamps = []
    for index, row in data.iterrows():
        try:
            ts = datetime.datetime(year=int(row['Year']),
                                month=int(row['Month']),
                                day=int(row['Day']),
                                hour=int(row['Hour']),
                                minute=int(row['Minute']), 
                                second=int(row['Second']))
        except ValueError as e:
            print(e)
            print(row)
            data = data.drop(index)
            continue
        timestamps.append(ts.timestamp())
    #data = data.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'])
    #print(data.shape)
    #print(len(timestamps))
    #data.insert(0, 'timestamp', timestamps)
    return data
