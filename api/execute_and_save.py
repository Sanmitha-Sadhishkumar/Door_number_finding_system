# execute_html.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def main():
    # Configure the headless browser
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    # Create a single instance of the web driver
    driver = webdriver.Chrome(options=chrome_options)

    # Open the modified HTML file in the headless browser
    driver.get(r"E:\Door_number_finding_system\api\index.html")

    # Wait for the page to load (adjust sleep time if needed)
    time.sleep(5)

    # Inject the JavaScript code for dynamic heading directly into the webpage
    js_code = """
        const headings = [0, 90, 180, 270];
        let currentHeadingIndex = 0;
        const streetViewImg = document.getElementById('streetViewImg');

        function displayNextStreetViewImage() {
            const heading = headings[currentHeadingIndex];
            const location = { lat: 11.3278082, lng: 77.7107402 };
            const streetViewUrl = `https://maps.googleapis.com/maps/api/streetview?size=800x600&location=${location.lat},${location.lng}&heading=${heading}&pitch=0&fov=70&key=AIzaSyB41DRUbKWJHPxaFjMAwdrzWzbVKartNGg`;
            streetViewImg.src = streetViewUrl;

            currentHeadingIndex = (currentHeadingIndex + 1) % headings.length;
        }

        // Start displaying images with a timed interval
        setInterval(displayNextStreetViewImage, 4000);
    """
    driver.execute_script(js_code)

    # Wait for the first screenshot to be captured (adjust sleep time if needed)
    time.sleep(5)

    # Capture four screenshots for each heading with 4 seconds interval
    for i in range(4):
        filename = f"E:/Door_number_finding_system/api/heading_{i+1}.png"
        capture_street_view_image(driver, filename)
        time.sleep(4)

    driver.quit()

def capture_street_view_image(driver, filename):
    # Capture a screenshot of the page
    driver.save_screenshot(filename)

if __name__ == "__main__":
    main()
