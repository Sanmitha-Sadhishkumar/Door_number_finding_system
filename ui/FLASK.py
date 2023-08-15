from flask import Flask, render_template, request
from gsv_and_geocoding import *
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_query = request.form['search_query']
        # You can perform any processing with the search query here

    return render_template("ui.html")

@app.route('/show-image', methods=['GET','POST'])
def show_image():
    search_query = request.form['search_query'].strip()  # Remove leading/trailing spaces
    print(search_query)
    g=geocode_address(search_query)
    if g:
        print(g.latitude,g.longitude)
        print("downloadinng gsv")
        execute_and_save(g.latitude,g.longitude)
    # Your existing logic here
    # ...

    return render_template('image.html')

if __name__ == '__main__':
    app.run()
