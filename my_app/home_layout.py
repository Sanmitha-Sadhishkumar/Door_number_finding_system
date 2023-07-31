from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.graphics import RoundedRectangle, Color
from kivy.core.window import Window
from kivy.graphics import RoundedRectangle, Color, Canvas, Ellipse
import sys
address = ''
sys.path.append(r"E:\git\Door_number_finding_system\api")
from gsv_and_geocoding import *


class TitleLayout(BoxLayout):
    pass


class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super(HomeScreen, self).__init__(**kwargs)

    def on_enter(self):
        layout = BoxLayout(orientation='vertical', pos_hint={'center_x': 0.5, 'center_y': 0.5}, spacing=15)

        self.address = TextInput(size_hint=(None, None),
                                 size=(200, 150),
                                 pos_hint={'center_x': 0.5, 'center_y': 0.5},
                                 background_color=(1, 0.47, 0.36, 1),  # #FF785D background
                                 foreground_color=(1, 1, 1, 1))  # White text color

        layout.add_widget(self.address)

        search = Button(text='Search',
                        size_hint=(None, None),
                        size=(100, 50),
                        padding=[100, 200],
                        pos_hint={'center_x': 0.5, 'center_y': 0.5},
                        background_color=(0.2, 0.2, 0.2, 1),  # Blackish grey background
                        color=(1, 1, 1, 1))  # White text color

        search.bind(on_press=self.on_press_button)
        layout.add_widget(search)

        popup = RoundedPopup(title='Enter the address:',
                         content=layout,
                         title_align='center',
                         size=(300, 310),
                         size_hint=(None, None),
                         separator_height=0,
                         auto_dismiss=False,
                         background_color=(1, 0.24, 0, 1))  # #FF3C00 (Shade of red) background for the popup
  # #FF3C00 (Shade of red) background for the popup

        search.bind(on_press=popup.dismiss)
        popup.open()

    def on_press_button(self, button):
        if self.address.text:
            self.address.text = self.address.text.replace('\n', ' ')
            print("\n\nAddress: ", self.address.text, '\n\n')
            loc = geocode_address(self.address.text)
            print("\n\nGeocoding result: ", loc, '\n\n')
            if loc:
                execute_and_save(loc.latitude,loc.longitude)
                # Switch to Screen 2 and pass the address to update the content
                app = App.get_running_app()
                app.root.current = 'display'
                app.root.get_screen('display').on_address_received(self.address.text)
        else:
            print("No address")


class DisplayScreen(Screen):
    def __init__(self, **kwargs):
        super(DisplayScreen, self).__init__(**kwargs)
        self.address_label = Label(text='', color=(0, 0, 0, 1))

    def on_enter(self):
        layout = BoxLayout(orientation='vertical')
        self.img = Image(source=r"E:\git\Door_number_finding_system\images\results\gsv_0.jpg")

        # Calculate the desired scale factor for the image (e.g., 2.0 for doubling the size)
        scale_factor = 1.2
        img_width = self.img.texture_size[0] * scale_factor
        img_height = self.img.texture_size[1] * scale_factor

        # Adjust the size of the image
        self.img.size_hint = (None, None)
        self.img.size = (img_width, img_height)

        layout.add_widget(self.img)

        # Calculate the size of the popup based on the image size with some margin
        popup_width = img_width + 40
        popup_height = img_height + 100

        popup = RoundedPopup(title=self.address_label.text,
                      content=layout,
                      title_align='center',
                      size=(popup_width, popup_height),
                      size_hint=(None, None),
                      separator_height=0,
                      auto_dismiss=False,
                      background_color=(1, 0.24, 0, 1))  # #FF3C00 (Shade of red) background for the popup

        popup.open()

    def on_address_received(self, address):
        # You can update the content of Screen 2 here based on the received address.
        # For example, you could update the image source based on the address:
        # self.img.source = "new_image_path.jpg"
        self.address_label.text = f"Address: {address}"

class RoundedPopup(Popup):
    def __init__(self, **kwargs):
        super(RoundedPopup, self).__init__(**kwargs)

    def draw_rounded_background(self):
        with self.canvas.before:
            # Draw a shadow behind the popup
            Color(0, 0, 0, 0.3)  # Transparent black
            Ellipse(pos=(self.x + 10, self.y - 10), size=(self.width + 20, self.height + 20))
            RoundedRectangle(pos=(self.x + 10, self.y - 10), size=(self.width + 20, self.height + 20), radius=[15, 15, 15, 15])
            
            # Draw a rounded background for the popup
            Color(1, 0.24, 0, 1)  # #FF3C00 (Shade of red) background color
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[15, 15, 15, 15])

    def on_size(self, *args):
        # Update the size of the rounded background when the popup size changes
        if hasattr(self, 'rect'):
            self.rect.size = self.size

    def on_pos(self, *args):
        # Update the position of the rounded background when the popup position changes
        if hasattr(self, 'rect'):
            self.rect.pos = self.pos

    def on_open(self):
        # This method will be called when the popup is about to be shown
        self.draw_rounded_background()

class HomeLayout(App):
    def build(self):
        self.screen_manager = ScreenManager()
        self.home_screen = HomeScreen(name='home')
        self.display_screen = DisplayScreen(name='display')
        self.screen_manager.add_widget(self.home_screen)
        self.screen_manager.add_widget(self.display_screen)
        return self.screen_manager


if __name__ == '__main__':
    app = HomeLayout()
    app.run()
