document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const generationForm = document.getElementById('generationForm');
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeImageBtn = document.getElementById('removeImage');
    const guidanceScaleSlider = document.getElementById('guidance_scale');
    const guidanceScaleValue = document.getElementById('guidance_scale_value');
    const stepsSlider = document.getElementById('steps');
    const stepsValue = document.getElementById('steps_value');
    const seedInput = document.getElementById('seed');
    const randomSeedBtn = document.getElementById('randomSeed');
    const generateBtn = document.getElementById('generateBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const resultsCard = document.getElementById('resultsCard');
    const inputImage = document.getElementById('inputImage');
    const outputImage = document.getElementById('outputImage');
    const resultPrompt = document.getElementById('resultPrompt');
    const resultGuidance = document.getElementById('resultGuidance');
    const resultSteps = document.getElementById('resultSteps');
    const resultSeed = document.getElementById('resultSeed');
    const downloadBtn = document.getElementById('downloadBtn');
    const newGenerationBtn = document.getElementById('newGenerationBtn');

    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropZone.classList.add('highlighted');
    }

    function unhighlight() {
        dropZone.classList.remove('highlighted');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            fileInput.files = files;
            updateImagePreview(files[0]);
        }
    }

    // Image preview functionality
    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            updateImagePreview(this.files[0]);
        }
    });

    function updateImagePreview(file) {
        // Check if file is an image
        if (!file.type.match('image.*')) {
            showNotification('Please select an image file (PNG or JPEG).', 'error');
            return;
        }

        const reader = new FileReader();
        
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            document.querySelector('.drop-zone-prompt').classList.add('hidden');
        };
        
        reader.readAsDataURL(file);
    }

    removeImageBtn.addEventListener('click', function() {
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        document.querySelector('.drop-zone-prompt').classList.remove('hidden');
    });

    // Parameter sliders
    guidanceScaleSlider.addEventListener('input', function() {
        guidanceScaleValue.textContent = this.value;
    });

    stepsSlider.addEventListener('input', function() {
        stepsValue.textContent = this.value;
    });

    // Random seed button
    randomSeedBtn.addEventListener('click', function() {
        const randomSeed = Math.floor(Math.random() * 1000000);
        seedInput.value = randomSeed;
    });

    // Form submission
    generationForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!validateForm()) {
            return;
        }
        
        const formData = new FormData(this);
        
        // Show loading overlay
        loadingOverlay.classList.remove('hidden');
        
        // Submit form
        fetch('/generate', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Generation failed');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading overlay
            loadingOverlay.classList.add('hidden');
            
            // Display results
            displayResults(data);
        })
        .catch(error => {
            // Hide loading overlay
            loadingOverlay.classList.add('hidden');
            
            showNotification(error.message, 'error');
        });
    });

    function validateForm() {
        // Check if prompt is entered
        const prompt = document.getElementById('prompt').value.trim();
        if (!prompt) {
            showNotification('Please enter a text prompt.', 'error');
            return false;
        }
        
        // Check if file is selected
        if (!fileInput.files.length) {
            showNotification('Please select a reference image.', 'error');
            return false;
        }
        
        return true;
    }

    function displayResults(data) {
        // Set images
        inputImage.src = data.input_image;
        outputImage.src = data.output_image;
        
        // Set parameters
        resultPrompt.textContent = data.parameters.prompt;
        resultGuidance.textContent = data.parameters.guidance_scale;
        resultSteps.textContent = data.parameters.steps;
        resultSeed.textContent = data.parameters.seed;
        
        // Show results card
        resultsCard.classList.remove('hidden');
        
        // Scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth' });
        
        // Setup download button
        downloadBtn.onclick = function() {
            downloadImage(data.output_image, 'generated-image.png');
        };
    }

    // Download image function
    function downloadImage(src, filename) {
        fetch(src)
            .then(response => response.blob())
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                a.remove();
            })
            .catch(error => {
                console.error('Download failed:', error);
                showNotification('Download failed. Please try again.', 'error');
            });
    }

    // New generation button
    newGenerationBtn.addEventListener('click', function() {
        resultsCard.classList.add('hidden');
        document.documentElement.scrollTop = 0;
    });

    // Notification function
    function showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        // Add icon based on type
        let icon = 'info-circle';
        if (type === 'error') icon = 'exclamation-circle';
        if (type === 'success') icon = 'check-circle';
        
        notification.innerHTML = `
            <i class="fas fa-${icon}"></i>
            <span>${message}</span>
            <button class="close-btn"><i class="fas fa-times"></i></button>
        `;
        
        // Style the notification
        Object.assign(notification.style, {
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            padding: '12px 15px',
            borderRadius: '4px',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            maxWidth: '400px',
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)',
            zIndex: '1000',
            animation: 'slideIn 0.3s ease-out forwards'
        });
        
        // Different colors based on type
        if (type === 'error') {
            notification.style.backgroundColor = '#f8d7da';
            notification.style.color = '#721c24';
            notification.style.border = '1px solid #f5c6cb';
        } else if (type === 'success') {
            notification.style.backgroundColor = '#d4edda';
            notification.style.color = '#155724';
            notification.style.border = '1px solid #c3e6cb';
        } else {
            notification.style.backgroundColor = '#d1ecf1';
            notification.style.color = '#0c5460';
            notification.style.border = '1px solid #bee5eb';
        }
        
        // Add close button functionality
        const closeBtn = notification.querySelector('.close-btn');
        closeBtn.style.backgroundColor = 'transparent';
        closeBtn.style.border = 'none';
        closeBtn.style.cursor = 'pointer';
        closeBtn.style.marginLeft = 'auto';
        
        closeBtn.addEventListener('click', function() {
            document.body.removeChild(notification);
        });
        
        // Add to DOM
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 5000);
    }

    // Add keyup listener to prompt for character count
    const promptInput = document.getElementById('prompt');
    const charCount = document.getElementById('charCount');
    
    if (promptInput && charCount) {
        promptInput.addEventListener('keyup', function() {
            const count = this.value.length;
            charCount.textContent = `${count}/500`;
            
            if (count > 500) {
                charCount.classList.add('text-danger');
            } else {
                charCount.classList.remove('text-danger');
            }
        });
    }

    // Add animation keyframes
    const styleSheet = document.createElement('style');
    styleSheet.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .hidden {
            display: none !important;
        }
        
        .highlighted {
            border-color: #4299e1 !important;
            background-color: rgba(66, 153, 225, 0.1) !important;
        }
    `;
    document.head.appendChild(styleSheet);

    // Initialize random seed on page load
    if (randomSeedBtn && seedInput) {
        randomSeedBtn.click();
    }

    // Add support for Enter key to submit when in prompt field
    if (promptInput) {
        promptInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                generationForm.dispatchEvent(new Event('submit'));
            }
        });
    }

    // Add hover effects for buttons
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.opacity = '0.9';
        });
        button.addEventListener('mouseleave', function() {
            this.style.opacity = '1';
        });
    });
    
    // Add loading spinner animation
    if (loadingOverlay) {
        const spinner = document.createElement('div');
        spinner.className = 'spinner';
        spinner.style.border = '5px solid rgba(255, 255, 255, 0.3)';
        spinner.style.borderRadius = '50%';
        spinner.style.borderTop = '5px solid #4299e1';
        spinner.style.width = '50px';
        spinner.style.height = '50px';
        spinner.style.animation = 'spin 1s linear infinite';
        
        const loadingText = document.createElement('div');
        loadingText.textContent = 'Generating image...';
        loadingText.style.marginTop = '15px';
        loadingText.style.color = 'white';
        loadingText.style.fontWeight = 'bold';
        
        const spinnerContainer = document.createElement('div');
        spinnerContainer.style.display = 'flex';
        spinnerContainer.style.flexDirection = 'column';
        spinnerContainer.style.alignItems = 'center';
        spinnerContainer.style.justifyContent = 'center';
        
        spinnerContainer.appendChild(spinner);
        spinnerContainer.appendChild(loadingText);
        loadingOverlay.appendChild(spinnerContainer);
        
        styleSheet.textContent += `
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
    }
});