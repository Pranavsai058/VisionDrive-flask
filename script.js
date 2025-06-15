document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const resultContainer = document.getElementById('resultContainer');
    const rgbInput = document.getElementById('rgbInput');
    const lidarInput = document.getElementById('lidarInput');
    const uploadBtn = form.querySelector('button[type="submit"]');

    // Initialize preview containers
    const rgbPreview = document.getElementById('rgbPreview');
    const lidarPreview = document.getElementById('lidarPreview');

    // Preview uploaded images
    rgbInput.addEventListener('change', function(e) {
        handleFileUpload(e.target, rgbPreview);
    });

    lidarInput.addEventListener('change', function(e) {
        handleFileUpload(e.target, lidarPreview);
    });

    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Validate inputs
        if (!validateInputs()) return;
        
        // Show loading state
        setLoadingState(true);
        
        try {
            const formData = new FormData(form);
            const response = await fetch('/', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const html = await response.text();
                updateResults(html);
            } else {
                throw new Error('Server returned an error');
            }
        } catch (error) {
            showError('Failed to process images. Please try again.');
            console.error('Error:', error);
        } finally {
            setLoadingState(false);
        }
    });

    // Helper functions
    function handleFileUpload(input, previewContainer) {
        const placeholder = previewContainer.querySelector('.preview-placeholder');
        
        if (input.files && input.files[0]) {
            // Remove existing preview if any
            const existingImg = previewContainer.querySelector('img');
            if (existingImg) existingImg.remove();
            
            // Remove existing remove button if any
            const existingBtn = previewContainer.querySelector('.remove-preview');
            if (existingBtn) existingBtn.remove();
            
            // Hide placeholder
            if (placeholder) placeholder.style.display = 'none';
            
            const reader = new FileReader();
            reader.onload = function(e) {
                // Create image element
                const img = document.createElement('img');
                img.src = e.target.result;
                img.alt = "Preview";
                
                // Create remove button
                const removeBtn = document.createElement('button');
                removeBtn.className = 'remove-preview';
                removeBtn.innerHTML = '×';
                removeBtn.setAttribute('data-for', input.id);
                
                // Add event to remove button
                removeBtn.addEventListener('click', function(e) {
                    e.preventDefault();
                    input.value = '';
                    img.remove();
                    removeBtn.remove();
                    if (placeholder) placeholder.style.display = 'flex';
                });
                
                // Append elements
                previewContainer.appendChild(img);
                previewContainer.appendChild(removeBtn);
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    function validateInputs() {
        if (!rgbInput.files[0] || !lidarInput.files[0]) {
            showError('Please upload both images!');
            return false;
        }
        return true;
    }

    function setLoadingState(isLoading) {
        if (isLoading) {
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<span class="spinner"></span> Processing...';
            resultContainer.innerHTML = `
                <div class="loading-state">
                    <div class="spinner"></div>
                    <p>Analyzing road conditions with enhanced edge detection...</p>
                </div>
            `;
        } else {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<span class="button-text">Analyze Road</span><span class="button-icon">→</span>';
        }
    }

    function updateResults(html) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const newResult = doc.getElementById('resultContainer').innerHTML;
        resultContainer.innerHTML = newResult;
        animateResults();
    }

    function animateResults() {
        const resultImages = document.querySelectorAll('.result-image');
        resultImages.forEach((img, index) => {
            img.style.opacity = '0';
            setTimeout(() => {
                img.style.transition = `opacity 0.5s ease ${index * 0.2}s`;
                img.style.opacity = '1';
            }, 100);
        });
    }

    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `
            <h3>Processing Error</h3>
            <p>${message}</p>
        `;
        resultContainer.innerHTML = '';
        resultContainer.appendChild(errorDiv);
    }
}); 