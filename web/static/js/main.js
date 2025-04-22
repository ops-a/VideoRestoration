document.addEventListener('DOMContentLoaded', () => {
    const uploadInput = document.getElementById('image-upload');
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');
    const originalImage = document.getElementById('original-image');
    const restoredImage = document.getElementById('restored-image');
    const metricsDiv = document.getElementById('metrics');

    uploadInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Show loading state
        loadingDiv.classList.remove('hidden');
        resultsDiv.classList.add('hidden');
        errorDiv.classList.add('hidden');

        // Create FormData
        const formData = new FormData();
        formData.append('file', file);

        try {
            // Upload image
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                // Display results
                originalImage.src = `/static/uploads/${data.original}`;
                restoredImage.src = `/static/uploads/${data.restored}`;
                
                // Display metrics
                metricsDiv.innerHTML = '';
                for (const [metric, value] of Object.entries(data.metrics)) {
                    const metricDiv = document.createElement('div');
                    metricDiv.className = 'bg-white p-2 rounded';
                    metricDiv.innerHTML = `
                        <div class="font-semibold">${metric}</div>
                        <div class="text-gray-600">${value.toFixed(4)}</div>
                    `;
                    metricsDiv.appendChild(metricDiv);
                }

                resultsDiv.classList.remove('hidden');
            } else {
                throw new Error(data.error || 'An error occurred');
            }
        } catch (error) {
            errorDiv.textContent = error.message;
            errorDiv.classList.remove('hidden');
        } finally {
            loadingDiv.classList.add('hidden');
        }
    });

    // Drag and drop functionality
    const dropZone = document.querySelector('label[for="image-upload"]');
    
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

    function highlight(e) {
        dropZone.classList.add('border-blue-500');
    }

    function unhighlight(e) {
        dropZone.classList.remove('border-blue-500');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        uploadInput.files = files;
        uploadInput.dispatchEvent(new Event('change'));
    }
}); 