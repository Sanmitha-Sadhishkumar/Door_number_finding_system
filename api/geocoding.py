from geopy.geocoders import Nominatim

locator = Nominatim(user_agent="myGeocoder")
# location = locator.geocode("Champ de Mars, Paris, France")
location = locator.geocode("Sardar Patel Rd, Guindy, Chennai, Tamil Nadu 600025")
print("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))
