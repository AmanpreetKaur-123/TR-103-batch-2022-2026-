// API endpoint - will be relative when served
const API_BASE_URL = ''; // Empty for same-origin requests

// DOM Elements
const form = document.getElementById('predictionForm');
const locationSelect = document.getElementById('location');
const areaTypeSelect = document.getElementById('areaType');
const availabilitySelect = document.getElementById('availability');
const resultDiv = document.getElementById('result');
const predictedPriceSpan = document.getElementById('predictedPrice');
const newPredictionBtn = document.getElementById('newPrediction');
const historyTable = document.getElementById('historyTable');

// State
let locations = [];
let areaTypes = [];
let availabilities = [];

// Toggle history section visibility
function setupHistoryToggle() {
    const historyToggle = document.getElementById('historyToggle');
    const historyCardBody = document.querySelector('#historyCard .card-body');
    const historyIcon = historyToggle.querySelector('i');
    
    historyToggle.addEventListener('click', () => {
        const isCollapsed = historyCardBody.classList.toggle('collapsed');
        historyToggle.classList.toggle('collapsed', isCollapsed);
        historyToggle.innerHTML = `<i class="fas ${isCollapsed ? 'fa-chevron-down' : 'fa-chevron-up'} me-1"></i>${isCollapsed ? 'Show' : 'Hide'}`;
    });
}

// Initialize the application
document.addEventListener('DOMContentLoaded', async () => {
    // Set up history toggle
    setupHistoryToggle();
    try {
        // Load initial data
        await Promise.all([
            loadLocations(),
            loadAreaTypes(),
            loadAvailabilities(),
            loadHistory()
        ]);
        
        // Set up event listeners
        setupEventListeners();
    } catch (error) {
        console.error('Error initializing application:', error);
        alert('Failed to initialize application. Please check the console for details.');
    }
});

// Set up event listeners
function setupEventListeners() {
    // Form submission
    form.addEventListener('submit', handleFormSubmit);
    
    // New prediction button
    newPredictionBtn.addEventListener('click', resetForm);
}

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();
    
    // Get form data
    const formData = {
        location: locationSelect.value,
        area_type: areaTypeSelect.value,
        availability: availabilitySelect.value,
        sqft: parseFloat(document.getElementById('sqft').value),
        bhk: parseInt(document.getElementById('bhk').value, 10),
        bathrooms: parseInt(document.getElementById('bath').value, 10)
    };
    
    try {
        // Show loading state
        const submitBtn = form.querySelector('button[type="submit"]');
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...';
        
        // Make prediction
        const prediction = await makePrediction(formData);
        
        // Display result
        displayResult(prediction, formData);
        
        // Reset button state
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalBtnText;
        
        // Reload history
        await loadHistory();
    } catch (error) {
        console.error('Prediction error:', error);
        alert(`Error making prediction: ${error.message}`);
        
        // Reset button state on error
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.disabled = false;
        submitBtn.innerHTML = 'Predict Price';
    }
}

// Make prediction API call
async function makePrediction(data) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to make prediction');
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Display prediction result
function displayResult(prediction, formData) {
    // Hide form, show result
    form.classList.add('d-none');
    resultDiv.classList.remove('d-none');
    
    // Format price with Indian Rupee symbol and commas
    const formattedPrice = new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        maximumFractionDigits: 0
    }).format(prediction.predicted_price);
    
    predictedPriceSpan.textContent = formattedPrice;
    
    // Scroll to result
    resultDiv.scrollIntoView({ behavior: 'smooth' });
}

// Reset form for a new prediction
function resetForm() {
    form.reset();
    resultDiv.classList.add('d-none');
    form.classList.remove('d-none');
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Load locations
async function loadLocations() {
    try {
        const response = await fetch(`${API_BASE_URL}/locations`);
        if (!response.ok) throw new Error('Failed to load locations');
        
        locations = await response.json();
        
        // Populate location dropdown
        locationSelect.innerHTML = '<option value="" selected disabled>Select Location</option>';
        locations.forEach(location => {
            const option = document.createElement('option');
            option.value = location;
            option.textContent = location;
            locationSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading locations:', error);
        throw error;
    }
}

// Load area types
async function loadAreaTypes() {
    try {
        const response = await fetch(`${API_BASE_URL}/area-types`);
        if (!response.ok) throw new Error('Failed to load area types');
        
        areaTypes = await response.json();
        
        // Populate area type dropdown
        areaTypeSelect.innerHTML = '<option value="" selected disabled>Select Area Type</option>';
        areaTypes.forEach(type => {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = type;
            areaTypeSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading area types:', error);
        throw error;
    }
}

// Load availabilities
async function loadAvailabilities() {
    try {
        const response = await fetch(`${API_BASE_URL}/availabilities`);
        if (!response.ok) throw new Error('Failed to load availabilities');
        
        availabilities = await response.json();
        
        // Populate availability dropdown
        availabilitySelect.innerHTML = '<option value="" selected disabled>Select Availability</option>';
        availabilities.forEach(availability => {
            const option = document.createElement('option');
            option.value = availability;
            option.textContent = availability;
            availabilitySelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading availabilities:', error);
        throw error;
    }
}

// Load prediction history
async function loadHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/history`);
        if (!response.ok) throw new Error('Failed to load history');
        
        const history = await response.json();
        
        // Clear existing rows
        historyTable.innerHTML = history.length === 0 
            ? '<tr><td colspan="6" class="text-center">No prediction history yet</td></tr>'
            : '';
        
        // Add history rows
        history.forEach(item => {
            const row = document.createElement('tr');
            
            // Format price
            const price = new Intl.NumberFormat('en-IN', {
                style: 'currency',
                currency: 'INR',
                maximumFractionDigits: 0
            }).format(parseFloat(item.predicted_price.replace(/[^0-9.]/g, '')));
            
            row.innerHTML = `
                <td>${item.location}</td>
                <td>${item.area_type}</td>
                <td>${Math.round(item.sqft).toLocaleString()}</td>
                <td>${item.bhk}</td>
                <td>${item.bathrooms}</td>
                <td class="fw-bold">${price}</td>
            `;
            
            historyTable.appendChild(row);
        });
    } catch (error) {
        console.error('Error loading history:', error);
        // Don't throw error to prevent blocking the app
        historyTable.innerHTML = '<tr><td colspan="6" class="text-center text-danger">Error loading history</td></tr>';
    }
}
