#AIzaSyB41DRUbKWJHPxaFjMAwdrzWzbVKartNGg
from google_streetview.api import *
from googlemaps import *
import os
from dotenv import load_dotenv
from geopy.geocoders import *
load_dotenv()
# execute_html.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def execute_and_save(lat,long):
    # Configure the headless browser
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    # Create a single instance of the web driver
    driver = webdriver.Chrome(options=chrome_options)

    # Open the modified HTML file in the headless browser
    driver.get(r"E:\git\Door_number_finding_system\api\index.html")

    # Wait for the page to load (adjust sleep time if needed)
    time.sleep(5)

    # Inject the JavaScript code for dynamic heading directly into the webpage
    js_code = f"""
    const headings = [0, 90, 180, 270];
    let currentHeadingIndex = 0;
    const streetViewImg = document.getElementById('streetViewImg');

    function displayNextStreetViewImage() {{
        const heading = headings[currentHeadingIndex];
        const location = {{ lat: {lat}, lng: {long} }};
        const streetViewUrl = 'https://maps.googleapis.com/maps/api/streetview?size=800x600&location=' + location.lat + ',' + location.lng + '&heading=' + heading + '&pitch=0&fov=70&key=AIzaSyB41DRUbKWJHPxaFjMAwdrzWzbVKartNGg';
        streetViewImg.src = streetViewUrl;

        currentHeadingIndex = (currentHeadingIndex + 1) % headings.length;
    }}

    // Start displaying images with a timed interval
    setInterval(displayNextStreetViewImage, 4000);
"""
    
    forward=f"""
    function moveForward() {{
        // Calculate the new latitude and longitude for moving forward
        const stepSize = 0.0001; // Adjust this value for larger or smaller steps
        const newLat = {lat} + stepSize;
        const newLong = {long};

        // Display the Street View image at the new location
        displayStreetViewImage(newLat, newLong);
    }}
    """

    backward=f"""
    function moveBackward() {{
        // Calculate the new latitude and longitude for moving backward
        const stepSize = -0.0001; // Adjust this value for larger or smaller steps
        const newLat = lat + stepSize;
        const newLong = long;

        // Display the Street View image at the new location
        displayStreetViewImage(newLat, newLong);
    }}
    """

    driver.execute_script(js_code)

    # Wait for the first screenshot to be captured (adjust sleep time if needed)
    time.sleep(5)

    # Capture four screenshots for each heading with 4 seconds interval
    for i in range(4):
        filename = f"E:/git/Door_number_finding_system/api/heading_{i+1}.png"
        capture_street_view_image(driver, filename)
        time.sleep(4)

    driver.quit()

def capture_street_view_image(driver, filename):
    # Capture a screenshot of the page
    driver.save_screenshot(filename)

def geocode_address(address:str):
    options.default_user_agent = "myGeocoder"
    locator = Photon(user_agent="myGeocoder")
    location = locator.geocode(address)
    if location:
        print("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))
        return location
    else:
        print("Can't do geocoding")
