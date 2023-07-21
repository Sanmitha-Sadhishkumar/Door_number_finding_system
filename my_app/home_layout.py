from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput

from kivy.uix.popup import Popup

class HomeLayout(App):
    def build(self):
        layout=Label(text="Easy Locator")
        return layout

    def on_start(self):
        layout = BoxLayout(orientation='vertical')

        self.address=TextInput(size_hint=(None,None),
                               pos_hint={'x':0.1,'y':.1},
                          size=(200,70))
        
        layout.add_widget(self.address)

        search=Button(text='Search',
                      size_hint=(None, None),
                      size=(100,50),
                      padding=[100,200],
                      pos_hint={'x':0.1,'y':.1})
        
        search.bind(on_press=self.on_press_button)
        layout.add_widget(search)

        popup = Popup(title='Enter the address : ',
                      content = layout,
                      size=(400,210),
                      size_hint=(None,None),
                      separator_height=0,
                      auto_dismiss=False) 
        popup.open()
        search.bind(on_press = popup.dismiss)

    def on_press_button(self, instance):
        if self.address.text:
            #self.address.text=self.address.text.split(',')
            print(self.address.text)
        else:
            print("no address")



if __name__ == '__main__':
    app = HomeLayout()
    app.run()