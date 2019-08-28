import folium
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from folium import plugins


def kml_data(Kml_Path):
    txt=open(Kml_Path,'r')
    datatxt=txt.read()
    datatxt=datatxt.split('coordinates')[-2]
    datatxt=datatxt.replace(' ','').split('\n')[1:-1]

    Datalatlon=[]
    for i in range(len(datatxt)):
        Datalatlon.append([float(datatxt[i].split(',')[-2]),float(datatxt[i].split(',')[-3])])

    Datalatlon=np.array(Datalatlon)
    lons_lats_vect = np.column_stack((Datalatlon[:,1], Datalatlon[:,0])) # Reshape coordinates
    return lons_lats_vect

def area_marker(Kml_Path):    
    lons_lats_vect=kml_data(Kml_Path)

    # point = Point(float(Data_to_save[500,5]), float(Data_to_save[500,6])) #(position_long,position_lat)
    # print(polygon.contains(point)) # check if polygon contains point
    # point

    geoJsonData = {
        "features": [
            {
                "geometry": {
                    "coordinates": lons_lats_vect.tolist(),
                    "type": "LineString"
                },
                "properties": {
                    "stroke": "#fc1717",
                    "stroke-opacity": 1,
                    "stroke-width": 3
                },
                "type": "Feature"
            },
        ],
        "type": "FeatureCollection"
    }
    return geoJsonData,lons_lats_vect


def Cluster_map_generation(Data_to_save,geoJsonData):
    m = folium.Map(location=[ float(Data_to_save[100,9]), float(Data_to_save[100,10])], zoom_start=15)
    
    folium.GeoJson(geoJsonData,
        style_function=lambda x: {
            'color' : x['properties']['stroke'],
            'weight' : x['properties']['stroke-width'],
            'opacity': 0.6,
            }).add_to(m)


    # instantiate a mark cluster object for the incidents in the dataframe
    incidents = plugins.MarkerCluster().add_to(m)

    # loop through the dataframe and add each data point to the mark cluster
    for i in range(0,len(Data_to_save)):
        folium.Marker(
            location=[float(Data_to_save[i,9]), float(Data_to_save[i,10])],
            icon=None,
        ).add_to(incidents)

    # display map
    m.save('Result/Map_Cluster.html')


def map_genration(Data_to_save,geoJsonData,lons_lats_vect):
    m = folium.Map(location=[ float(Data_to_save[100,9]), float(Data_to_save[100,10])], zoom_start=15)
    ## Loading KML file
    folium.GeoJson(geoJsonData,
        style_function=lambda x: {
            'color' : x['properties']['stroke'],
            'weight' : x['properties']['stroke-width'],
            'opacity': 0.6,
            }).add_to(m);

    polygon = Polygon(lons_lats_vect) # create polygon

    # instantiate a feature group for the incidents in the dataframe
    incidents = folium.map.FeatureGroup()

    tok=None
    Time_Saving=[]
    Time_Saving.append(['Activity Starting Time',Data_to_save[0,1]])
    # print('Activity Starting Time:',Data_to_save[0,1])
    for i in range(0,len(Data_to_save)):
        point = Point(float(Data_to_save[i,10]),float(Data_to_save[i,9])) #(position_long,position_lat)


        if Data_to_save[i,-1]=='0':
            if polygon.contains(point):
                if tok!=0:
    #                 print ('Lowest start inside:',Data_to_save[i,1])
                    Time_Saving.append(['Lowest start inside:',Data_to_save[i,1]])
                    tok=0
                    Entering_Time_index=i

                incidents.add_child(
                    folium.features.CircleMarker(
                        [float(Data_to_save[i,9]), float(Data_to_save[i,10])],
                        radius=3, # define how big you want the circle markers to be
                        color='yellow',
                        fill=True,
                        popup=Data_to_save[i,1],
                        fill_color='blue',
                        fill_opacity=0.6
                    )
                )
            else:

                if tok!=0:
    #                 print ('Lowest start outside:',Data_to_save[i,1])
                    Time_Saving.append(['Lowest start outside:',Data_to_save[i,1]])
                    tok=0
                    Entering_Time_index=i

                incidents.add_child(
                    folium.features.CircleMarker(
                        [float(Data_to_save[i,9]), float(Data_to_save[i,10])],
                        radius=3, # define how big you want the circle markers to be
                        color='red',
                        fill=True,
                        popup=Data_to_save[i,1],
                        fill_color='yellow',
                        fill_opacity=0.6
                    )
                )


        if Data_to_save[i,-1]=='5': 

            if polygon.contains(point):
                if tok!=1:
    #                 print ('Medium start inside:',Data_to_save[i,1])
                    Time_Saving.append(['Medium start inside:',Data_to_save[i,1]])
                    tok=1
                    Outside_Time_index=i

                incidents.add_child(
                    folium.features.CircleMarker(
                        [float(Data_to_save[i,9]), float(Data_to_save[i,10])],
                        radius=3, # define how big you want the circle markers to be
                        color='green',
                        fill=True,
                        popup=Data_to_save[i,1],
                        fill_color='blue',
                        fill_opacity=0.6
                    )
                )
            else:            
                if tok!=1:
    #                 print ('Medium start outside:',Data_to_save[i,1])
                    Time_Saving.append(['Medium start outside:',Data_to_save[i,1]])
                    tok=1
                    Outside_Time_index=i

                incidents.add_child(
                    folium.features.CircleMarker(
                        [float(Data_to_save[i,9]), float(Data_to_save[i,10])],
                        radius=3, # define how big you want the circle markers to be
                        color='red',
                        fill=True,
                        popup=Data_to_save[i,1],
                        fill_color='green',
                        fill_opacity=0.6
                    )
                )


        if Data_to_save[i,-1]=='10': 
            if polygon.contains(point):
                if tok!=2:
    #                 print ('Highest start inside:',Data_to_save[i,1])
                    Time_Saving.append(['Highest start inside:',Data_to_save[i,1]])
                    tok=2
                    Outside_Time_index=i

                incidents.add_child(
                    folium.features.CircleMarker(
                        [float(Data_to_save[i,9]), float(Data_to_save[i,10])],
                        radius=3, # define how big you want the circle markers to be
                        color='red',
                        fill=True,
                        popup=Data_to_save[i,1],
                        fill_color='blue',
                        fill_opacity=0.6
                    )
                )
            else:
                if tok!=2:
    #                 print ('Highest start outside:',Data_to_save[i,1])
                    Time_Saving.append(['Highest start outside:',Data_to_save[i,1]])
                    tok=2
                    Outside_Time_index=i

                incidents.add_child(
                    folium.features.CircleMarker(
                        [float(Data_to_save[i,9]), float(Data_to_save[i,10])],
                        radius=3, # define how big you want the circle markers to be
                        color='red',
                        fill=True,
                        popup=Data_to_save[i,1],
                        fill_color='red',
                        fill_opacity=0.6
                    )
                )

    # print('Activity End Time:',Data_to_save[-1,1])
    Time_Saving.append(['Activity End Time',Data_to_save[-1,1]])

    m.add_child(incidents);
    m.save('Result/Map.html')
    return Time_Saving