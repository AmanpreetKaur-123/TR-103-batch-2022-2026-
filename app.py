import BangalorePricePrediction as tm
import pandas as pd
from datetime import datetime
import os
import sys

# Constants
DATA_FILE = "predictions_history.csv"
HISTORY_COLUMNS = ["timestamp", "location", "area_type", "availability", "sqft", "bhk", "bathrooms", "predicted_price"]

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_header(title):
    """Display a formatted header"""
    clear_screen()
    print("=" * 50)
    print(f"{' ' * ((50 - len(title)) // 2)}{title}")
    print("=" * 50)

def display_menu():
    """Display the main menu and get user choice"""
    display_header("House Price Prediction")
    print("\nMain Menu:")
    print("1. View available locations")
    print("2. View area types")
    print("3. View availability options")
    print("4. Predict house price")
    print("5. View prediction history")
    print("6. Exit")
    
    while True:
        choice = input("\nSelect an option (1-6): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6']:
            return choice
        print("Invalid choice. Please enter a number between 1-6.")

def get_float_input(prompt, min_val=0):
    """Get a valid float input from user"""
    while True:
        try:
            value = float(input(prompt))
            if value >= min_val:
                return value
            print(f"Value must be at least {min_val}.")
        except ValueError:
            print("Please enter a valid number.")

def get_int_input(prompt, min_val=1, max_val=None):
    """Get a valid integer input from user"""
    while True:
        try:
            value = int(input(prompt))
            if value < min_val or (max_val is not None and value > max_val):
                if max_val:
                    print(f"Please enter a number between {min_val} and {max_val}.")
                else:
                    print(f"Value must be at least {min_val}.")
                continue
            return value
        except ValueError:
            print("Please enter a whole number.")

def save_prediction(prediction_data):
    """Save prediction to history file"""
    try:
        df = pd.DataFrame([prediction_data])
        if not os.path.exists(DATA_FILE):
            df.to_csv(DATA_FILE, index=False)
        else:
            df.to_csv(DATA_FILE, mode='a', header=False, index=False)
    except Exception as e:
        print(f"Warning: Could not save prediction history: {e}")

def view_prediction_history():
    """Display prediction history"""
    try:
        if not os.path.exists(DATA_FILE):
            input("\nNo prediction history found. Press Enter to continue...")
            return
            
        df = pd.read_csv(DATA_FILE)
        if df.empty:
            input("\nNo prediction history found. Press Enter to continue...")
            return
            
        display_header("Prediction History")
        print(df.to_string(index=False))
        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"\nError viewing history: {e}")
        input("Press Enter to continue...")

def predict_price():
    """Handle the price prediction flow"""
    display_header("Predict House Price")
    
    try:
        # Display available options
        print("\nAvailable locations (first 10 shown):")
        locations = tm.get_location_names()
        for i, loc in enumerate(locations[:10], 1):
            print(f"{i}. {loc}")
        if len(locations) > 10:
            print(f"... and {len(locations) - 10} more locations available")
        
        # Get user input with validation
        location = input("\nEnter location: ").strip()
        if location not in locations:
            print(f"Warning: '{location}' not found in our database. Using closest match.")
        
        print("\nAvailable area types:")
        area_types = tm.get_area_values()
        for i, area in enumerate(area_types, 1):
            print(f"{i}. {area}")
        area_type = input("\nSelect area type (enter number): ")
        try:
            area_type = area_types[int(area_type) - 1]
        except (ValueError, IndexError):
            area_type = area_type  # Use as is if not a valid number
        
        print("\nAvailability options:")
        availabilities = tm.get_availability_values()
        for i, avail in enumerate(availabilities, 1):
            print(f"{i}. {avail}")
        availability = input("\nSelect availability (enter number): ")
        try:
            availability = availabilities[int(availability) - 1]
        except (ValueError, IndexError):
            availability = availability  # Use as is if not a valid number
        
        sqft = get_float_input("\nEnter total square feet: ", min_val=100)
        bhk = get_int_input("Enter number of BHK: ", min_val=1)
        bathrooms = get_int_input("Enter number of bathrooms: ", min_val=1)
        
        # Make prediction
        prediction = round(float(tm.predict_house_price(
            location, area_type, availability, sqft, bhk, bathrooms
        )), 2)
        
        # Display result
        print(f"\n{'='*30}")
        print(f"Estimated Price: ₹{prediction:,.2f}")
        print(f"{'='*30}")
        
        # Save to history
        save_prediction({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "location": location,
            "area_type": area_type,
            "availability": availability,
            "sqft": sqft,
            "bhk": bhk,
            "bathrooms": bathrooms,
            "predicted_price": f"₹{prediction:,.2f}"
        })
        
        input("\nPress Enter to return to the main menu...")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        input("\nPress Enter to continue...")

def run_demo():
    """Run a demo of the house price prediction"""
    print("\n=== Running Demo ===\n")
    
    # Display some locations
    print("Sample locations:")
    locations = tm.get_location_names()
    for loc in locations[:5]:
        print(f"- {loc}")
    print("... and more")
    
    # Display area types
    print("\nArea types:")
    for area in tm.get_area_values():
        print(f"- {area}")
    
    # Make a sample prediction
    print("\nMaking a sample prediction for:")
    print("Location: 1st Phase JP Nagar")
    print("Area Type: Built-up  Area")
    print("Availability: Ready To Move")
    print("Square Feet: 1000")
    print("BHK: 2")
    print("Bathrooms: 2")
    
    try:
        prediction = tm.predict_house_price(
            "1st Phase JP Nagar", "Built-up  Area", "Ready To Move", 1000, 2, 2
        )
        print(f"\nPredicted Price: ₹{prediction:,.2f}")
    except Exception as e:
        print(f"\nError making prediction: {e}")
    
    print("\nDemo complete! You can run the full interactive version with: python app.py")

def main():
    # Check if running in non-interactive mode
    if not sys.stdin.isatty():
        run_demo()
        return
        
    # Load saved attributes
    try:
        tm.load_saved_attributes()
    except Exception as e:
        print(f"Error loading model: {e}")
        input("Press Enter to exit...")
        return
    
    while True:
        try:
            choice = display_menu()
            
            if choice == '1':
                display_header("Available Locations")
                locations = tm.get_location_names()
                for i, loc in enumerate(locations, 1):
                    print(f"{i}. {loc}")
                input(f"\nTotal {len(locations)} locations found. Press Enter to continue...")
                
            elif choice == '2':
                display_header("Area Types")
                for i, area in enumerate(tm.get_area_values(), 1):
                    print(f"{i}. {area}")
                input("\nPress Enter to continue...")
                
            elif choice == '3':
                display_header("Availability Options")
                for i, avail in enumerate(tm.get_availability_values(), 1):
                    print(f"{i}. {avail}")
                input("\nPress Enter to continue...")
                
            elif choice == '4':
                predict_price()
                
            elif choice == '5':
                view_prediction_history()
                
            elif choice == '6':
                print("\nThank you for using House Price Prediction. Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        input("Press Enter to exit...")