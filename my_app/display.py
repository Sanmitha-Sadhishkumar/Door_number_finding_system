from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

class DisplayImage(App):
    def __inti__(self,address):
        self.address=address

    def build(self):
        layout=BoxLayout(orientation='vertical')
        img=Image(source=r"E:\git\Door_number_finding_system\images\results\gsv_0.jpg",
                  )
        return img

DisplayImage().run()