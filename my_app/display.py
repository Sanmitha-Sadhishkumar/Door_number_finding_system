from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
import sys

sys.path.append(r"E:\git\Door_number_finding_system\api")
from gsv_and_geocoding import *

class DisplayScreen(Screen):
    def __init__(self, **kwargs):
        super(DisplayScreen, self).__init__(**kwargs)

    def on_enter(self):
        self.layout = BoxLayout(orientation='vertical')
        self.img = Image(source=r"E:\git\Door_number_finding_system\images\results\gsv_0.jpg")
        self.layout.add_widget(self.img)
        self.add_widget(self.layout)

class DisplayImageApp(App):
    def __init__(self, address):
        super(DisplayImageApp, self).__init__()
        self.address = address

    def build(self):
        self.sm = ScreenManager()
        self.display_screen = DisplayScreen(name='display')
        self.sm.add_widget(self.display_screen)
        return self.sm

    def on_start(self):
        # Perform any initialization tasks here if needed.
        pass

    def on_stop(self):
        # Perform any cleanup tasks here if needed.
        pass

    def on_pause(self):
        # Return True to pause the app when it goes to the background.
        return True

    def on_resume(self):
        # Resume the app from the background.
        pass

    def on_address_received(self, address):
        # This method will be called when the address is received from Screen 1.
        # You can update the content of Screen 2 here if needed.
        self.display_screen.address = address
        # For example, you could update the image source based on the address:
        # self.display_screen.img.source = "new_image_path.jpg"

# Run the app with the address received from Screen 1
if __name__ == '__main__':
    address = "Default Address"  # You can set the default address or leave it empty
    app = DisplayImageApp(address)
    app.run()
