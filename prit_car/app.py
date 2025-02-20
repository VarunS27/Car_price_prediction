from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login/<user_type>')
def login(user_type):
    return render_template('login.html', user_type=user_type)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/car')
def car():
    return render_template('cars.html')

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/add")
def add():
    return render_template("add.html")

if __name__ == '__main__':
    app.run(debug=True)
