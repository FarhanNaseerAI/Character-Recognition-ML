document.addEventListener('DOMContentLoaded', () => {
    // --- State & DOM Elements ---
    let currentMode = 'draw'; // 'draw' or 'upload'
    let selectedFile = null;

    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // Canvas
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clear-btn');
    const recognizeBtn = document.getElementById('recognize-btn');
    const canvasPlaceholder = document.getElementById('canvas-placeholder');

    // Upload
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const analyzeUploadBtn = document.getElementById('analyze-upload-btn');
    const clearUploadBtn = document.getElementById('clear-upload-btn');

    // Results
    const stateEmpty = document.getElementById('empty-state');
    const stateLoading = document.getElementById('loading-state');
    const stateResult = document.getElementById('result-state');
    const predictedDigit = document.getElementById('predicted-digit');
    const predictedConfidence = document.getElementById('predicted-confidence');
    const topPredictionsList = document.getElementById('top-predictions');

    // History
    const historyList = document.getElementById('history-list');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    // Canvas Setup
    // Initialize canvas with white background
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Drawing State
    let isDrawing = false;
    let hasDrawn = false;
    
    // Configure context for drawing
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'black';

    // --- Tab Switching ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            currentMode = tabId;

            // Update buttons
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update content
            tabContents.forEach(c => c.classList.remove('active'));
            document.getElementById(`${tabId}-tab`).classList.add('active');

            // Reset view
            showState('empty');
        });
    });

    // --- Canvas Drawing Logic ---
    function startDrawing(e) {
        isDrawing = true;
        hasDrawn = true;
        canvasPlaceholder.style.opacity = '0';
        draw(e);
    }

    function stopDrawing() {
        isDrawing = false;
        ctx.beginPath(); // Reset path to prevent connecting dots
    }

    function draw(e) {
        if (!isDrawing) return;

        const rect = canvas.getBoundingClientRect();
        
        // Handle both mouse and touch events
        let clientX = e.clientX;
        let clientY = e.clientY;
        
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
            e.preventDefault(); // Prevent scrolling while touching canvas
        }

        const x = clientX - rect.left;
        const y = clientY - rect.top;

        // Scale coordinates if canvas display size differs from internal size
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        ctx.lineTo(x * scaleX, y * scaleY);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x * scaleX, y * scaleY);
    }

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDrawing);

    clearBtn.addEventListener('click', () => {
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        hasDrawn = false;
        canvasPlaceholder.style.opacity = '1';
        showState('empty');
    });

    // --- Upload Logic ---
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!allowedTypes.includes(file.type)) {
            showToast('Invalid file type. Please upload a PNG or JPG.');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            uploadPlaceholder.style.display = 'none';
            analyzeUploadBtn.disabled = false;
            clearUploadBtn.style.display = 'block';
            showState('empty');
        };
        reader.readAsDataURL(file);
    }

    clearUploadBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent opening file dialog
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        imagePreview.style.display = 'none';
        uploadPlaceholder.style.display = 'flex';
        analyzeUploadBtn.disabled = true;
        clearUploadBtn.style.display = 'none';
        showState('empty');
    });

    // --- API Communication ---
    function showState(state) {
        [stateEmpty, stateLoading, stateResult].forEach(el => el.classList.remove('active'));
        
        if (state === 'empty') stateEmpty.classList.add('active');
        if (state === 'loading') stateLoading.classList.add('active');
        if (state === 'result') stateResult.classList.add('active');
    }

    function updateResults(data) {
        const label = data.character || data.prediction;
        predictedDigit.textContent = label;
        predictedConfidence.textContent = `${data.confidence}%`;
        
        topPredictionsList.innerHTML = '';
        data.top_predictions.forEach((pred, index) => {
            const barWidth = Math.max(pred.confidence, 1);
            const predLabel = pred.character || pred.digit;
            
            const itemHTML = `
                <div class="pred-item">
                    <div class="pred-digit">${predLabel}</div>
                    <div class="pred-bar-container">
                        <div class="pred-bar" style="width: 0%; opacity: ${index === 0 ? 1 : 0.6}" data-width="${barWidth}%"></div>
                    </div>
                    <div class="pred-val">${pred.confidence}%</div>
                </div>
            `;
            topPredictionsList.insertAdjacentHTML('beforeend', itemHTML);
        });

        // Trigger reflow for animation
        setTimeout(() => {
            document.querySelectorAll('.pred-bar').forEach(bar => {
                bar.style.width = bar.getAttribute('data-width');
            });
        }, 50);

        addToHistory(label, data.confidence);
    }

    recognizeBtn.addEventListener('click', async () => {
        if (!hasDrawn) {
            showToast('Please draw a character first.');
            return;
        }

        showState('loading');
        recognizeBtn.disabled = true;

        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'canvas.png');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    updateResults(data);
                    showState('result');
                } else {
                    showToast(data.error || 'Failed to predict character');
                    showState('empty');
                }
            } catch (error) {
                console.error(error);
                showToast('Network error occurred.');
                showState('empty');
            } finally {
                recognizeBtn.disabled = false;
            }
        }, 'image/png');
    });

    analyzeUploadBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        showState('loading');
        analyzeUploadBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict-upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                updateResults(data);
                showState('result');
            } else {
                showToast(data.error || 'Failed to analyze image');
                showState('empty');
            }
        } catch (error) {
            console.error(error);
            showToast('Network error occurred.');
            showState('empty');
        } finally {
            analyzeUploadBtn.disabled = false;
        }
    });

    // --- History Logic ---
    function addToHistory(digit, confidence) {
        const emptyMsg = historyList.querySelector('.history-empty');
        if (emptyMsg) {
            emptyMsg.remove();
        }

        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.innerHTML = `
            <div class="hist-left">
                <div class="hist-digit">${digit}</div>
                <div class="hist-conf">${confidence}%</div>
            </div>
            <div class="hist-time">${timeStr}</div>
        `;

        historyList.prepend(historyItem);
    }

    clearHistoryBtn.addEventListener('click', () => {
        historyList.innerHTML = '<div class="history-empty">No predictions yet</div>';
    });

    fetch('/health')
        .then(response => response.json())
        .then(data => {
            if (!data.model_loaded) {
                showToast('Model not loaded. Train it with: python training/train_model.py');
            }
        })
        .catch(() => {});

    // --- Toast Notifications ---
    function showToast(message) {
        let toast = document.querySelector('.toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.className = 'toast';
            document.body.appendChild(toast);
        }
        
        toast.textContent = message;
        toast.classList.add('show');
        
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }
});
