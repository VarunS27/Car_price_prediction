from flask import Flask, render_template
import requests

app = Flask(__name__)

PIXABAY_API_KEY = '48956942-b8756c0f5361200fb2260a422'  # Replace with your Pixabay API key


def fetch_images():
    images = []
    url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q=cars&image_type=photo&per_page=3"
    response = requests.get(url)

    print(f"Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")  # Print API response

    if response.status_code == 200:
        try:
            data = response.json()
            print(f"Response Data: {data}")  # Print parsed data

            if 'hits' in data and len(data['hits']) > 0:
                for hit in data['hits']:
                    image_url = hit.get("webformatURL", "https://via.placeholder.com/800")
                    images.append(image_url)
            else:
                images.append("https://via.placeholder.com/800")
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            images.append("https://via.placeholder.com/800")
    else:
        print(f"Error fetching images: {response.status_code} - {response.text}")
        images.append("https://via.placeholder.com/800")

    return images


@app.route('/')
def home():
    images = fetch_images()  # Fetch images when rendering home page
    return render_template('index.html', images=images)

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
