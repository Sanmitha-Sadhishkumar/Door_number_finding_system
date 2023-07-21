from geopy.geocoders import Nominatim
from gsv_and_geocoding import *

locator = Nominatim(user_agent="myGeocoder")
# location = locator.geocode("Champ de Mars, Paris, France")
location = locator.geocode("Surampatti, Erode")
print("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))
