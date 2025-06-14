// script.js

// --- Functions for the Home Page ---
document.addEventListener('DOMContentLoaded', function() {
    const queryForm = document.querySelector('form[method="POST"]');
    if (queryForm) {
        queryForm.addEventListener('submit', function(event) {
            const queryInput = queryForm.querySelector('input[name="query"]');
            if (queryInput && queryInput.value.trim() === '') {
                alert('Please enter a question before submitting.');
                event.preventDefault(); // Prevent form submission
            }
        });
    }

    const logoutLink = document.querySelector('a[href*="/logout"]');
    if (logoutLink) {
        logoutLink.addEventListener('click', function(event) {
            if (!confirm('Are you sure you want to log out?')) {
                event.preventDefault(); // Prevent navigation
            }
        });
    }
});

// --- Functions for the Input New Data Page ---
document.addEventListener('DOMContentLoaded', function() {
    const newDataForm = document.querySelector('form[method="POST"] textarea[name="new_data"]');
    if (newDataForm) {
        newDataForm.closest('form').addEventListener('submit', function(event) {
            if (newDataForm.value.trim() === '') {
                alert('Please enter some data before storing.');
                event.preventDefault(); // Prevent form submission
            }
        });
    }
});