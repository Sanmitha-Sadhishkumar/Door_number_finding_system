from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.app import App
from home_layout import *
from display import *

class Home(ScreenManager):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(HomeLayout())
        sm.add_widget(DisplayImage())
        return sm
    
if __name__=='__main__':
    H=Home()
    H.run()
