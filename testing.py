from optparse import OptionParser

from active_utils import fitparser
from active_utils import Model_m
from active_utils import maper

from keras.models import load_model

parser = OptionParser()

parser.add_option("--file_path", "--fp", dest="File_path", help="Path to Garmin Activity .fit file")
parser.add_option("--kml_path","--kp",  dest="Kml_Path", help="Path to Google map generated KML file for area")
parser.add_option("--model_path", "--mp", dest="Model_Path", help="Path to trained model file",default='MODELS/CNNLSTM.h5')
parser.add_option("--std_path", "--sp", dest="Standard_scalar_path", help="Path to saved standard_scalar file",default='MODELS/CNNLSTM_std_scaler.bin')



(options, args) = parser.parse_args()

print ('Please Wait Program running ...')
model=load_model(options.Model_Path)

data=fitparser.fitparser(options.File_path)

Data_to_save=Model_m.Model_Result(model,data,options.Standard_scalar_path)

geoJsonData,lons_lats_vect=maper.area_marker(options.Kml_Path)

Time_Saving=maper.map_genration(Data_to_save,geoJsonData,lons_lats_vect)

Model_m.summary_saving(Time_Saving,options.File_path)

maper.Cluster_map_generation(Data_to_save,geoJsonData)
print ('Program completed all the results are saved in "Result/" directory')