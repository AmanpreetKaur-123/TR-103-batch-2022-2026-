import http.server
import socketserver
import json
import os
from datetime import datetime
import BangalorePricePrediction as tm

# Initialize the model
tm.load_saved_attributes()

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)
HISTORY_FILE = 'data/predictions.json'

# Load prediction history
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

# Save prediction to history
def save_to_history(data):
    history = load_history()
    history.insert(0, data)
    # Keep only the last 50 predictions
    history = history[:50]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

# Custom request handler
class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/locations':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(tm.get_location_names()).encode())
        elif self.path == '/area-types':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(tm.get_area_values()).encode())
        elif self.path == '/availabilities':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(tm.get_availability_values()).encode())
        elif self.path == '/history':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(load_history()).encode())
        else:
            # Serve static files
            if self.path == '/':
                self.path = '/index.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            try:
                # Make prediction
                prediction = tm.predict_house_price(
                    data['location'],
                    data['area_type'],
                    data['availability'],
                    float(data['sqft']),
                    int(data['bhk']),
                    int(data['bathrooms'])
                )
                
                # Prepare response
                response = {
                    'status': 'success',
                    'predicted_price': float(prediction)
                }
                
                # Save to history
                history_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'location': data['location'],
                    'area_type': data['area_type'],
                    'availability': data['availability'],
                    'sqft': float(data['sqft']),
                    'bhk': int(data['bhk']),
                    'bathrooms': int(data['bathrooms']),
                    'predicted_price': f"₹{float(prediction):,.2f}"
                }
                save_to_history(history_entry)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'error',
                    'error': str(e)
                }).encode())
        else:
            self.send_response(404)
            self.end_headers()

# Set up and start the server
def run(port=8000):
    handler = RequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at port {port}")
        print(f"Open http://localhost:{port} in your browser")
        httpd.serve_forever()

if __name__ == "__main__":
    run()