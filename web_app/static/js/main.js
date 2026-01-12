/**
 * Skin Lesion Classifier - Frontend JavaScript
 * Handles image upload, preview, and API communication
 */

document.addEventListener("DOMContentLoaded", function () {
  // DOM Elements
  const uploadArea = document.getElementById("uploadArea");
  const uploadPlaceholder = document.getElementById("uploadPlaceholder");
  const imageInput = document.getElementById("imageInput");
  const previewImage = document.getElementById("previewImage");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const clearBtn = document.getElementById("clearBtn");
  const loadingState = document.getElementById("loadingState");
  const resultsSection = document.getElementById("resultsSection");

  // Result elements
  const mainPrediction = document.getElementById("mainPrediction");
  const severityBadge = document.getElementById("severityBadge");
  const predictionName = document.getElementById("predictionName");
  const predictionCode = document.getElementById("predictionCode");
  const confidenceValue = document.getElementById("confidenceValue");
  const confidenceFill = document.getElementById("confidenceFill");
  const predictionDescription = document.getElementById(
    "predictionDescription"
  );
  const predictionsList = document.getElementById("predictionsList");

  // State
  let selectedFile = null;

  // ========================================
  // Event Listeners
  // ========================================

  // Click to upload
  uploadArea.addEventListener("click", () => {
    imageInput.click();
  });

  // File input change
  imageInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileSelect(file);
    }
  });

  // Drag and drop events
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      handleFileSelect(file);
    }
  });

  // Analyze button
  analyzeBtn.addEventListener("click", analyzeImage);

  // Clear button
  clearBtn.addEventListener("click", clearSelection);

  // ========================================
  // Functions
  // ========================================

  /**
   * Handle file selection and show preview
   */
  function handleFileSelect(file) {
    // Validate file type
    const validTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!validTypes.includes(file.type)) {
      alert("Please select a valid image file (JPG, PNG)");
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert("File size too large. Please select an image under 10MB.");
      return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewImage.classList.remove("hidden");
      uploadPlaceholder.classList.add("hidden");
      uploadArea.classList.add("has-image");
    };
    reader.readAsDataURL(file);

    // Enable buttons
    analyzeBtn.disabled = false;
    clearBtn.disabled = false;

    // Hide previous results
    resultsSection.classList.add("hidden");
  }

  /**
   * Clear selected image and reset UI
   */
  function clearSelection() {
    selectedFile = null;
    imageInput.value = "";
    previewImage.src = "";
    previewImage.classList.add("hidden");
    uploadPlaceholder.classList.remove("hidden");
    uploadArea.classList.remove("has-image");
    analyzeBtn.disabled = true;
    clearBtn.disabled = true;
    resultsSection.classList.add("hidden");
    loadingState.classList.add("hidden");
  }

  /**
   * Send image to API and display results
   */
  async function analyzeImage() {
    if (!selectedFile) {
      alert("Please select an image first");
      return;
    }

    // Show loading state
    loadingState.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    analyzeBtn.disabled = true;

    try {
      // Prepare form data
      const formData = new FormData();
      formData.append("image", selectedFile);

      // Send to API
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to analyze image");
      }

      // Display results
      displayResults(data);
    } catch (error) {
      console.error("Error:", error);
      alert("Error analyzing image: " + error.message);
    } finally {
      loadingState.classList.add("hidden");
      analyzeBtn.disabled = false;
    }
  }

  /**
   * Display prediction results
   */
  function displayResults(data) {
    const prediction = data.prediction;
    const allPredictions = data.all_predictions;

    // Update main prediction card
    predictionName.textContent = prediction.class_name;
    predictionCode.textContent = `Code: ${prediction.class_code.toUpperCase()}`;
    predictionDescription.textContent = prediction.description;

    // Update severity badge
    severityBadge.textContent = prediction.severity;
    severityBadge.className =
      "prediction-badge " + getSeverityClass(prediction.severity);

    // Animate confidence meter
    confidenceValue.textContent = `${prediction.confidence.toFixed(1)}%`;
    setTimeout(() => {
      confidenceFill.style.width = `${prediction.confidence}%`;
    }, 100);

    // Update all predictions list
    predictionsList.innerHTML = "";
    allPredictions.forEach((pred, index) => {
      const item = createPredictionItem(pred, index);
      predictionsList.appendChild(item);
    });

    // Show results section
    resultsSection.classList.remove("hidden");

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /**
   * Create a prediction list item element
   */
  function createPredictionItem(pred, index) {
    const item = document.createElement("div");
    item.className = "prediction-item" + (index === 0 ? " top" : "");

    item.innerHTML = `
            <span class="rank">${index + 1}</span>
            <span class="name">${pred.class_name}</span>
            <span class="code">(${pred.class_code})</span>
            <span class="probability">${pred.probability.toFixed(1)}%</span>
            <div class="bar-container">
                <div class="bar-fill" style="width: ${
                  pred.probability
                }%; background-color: ${pred.color}"></div>
            </div>
        `;

    return item;
  }

  /**
   * Get CSS class based on severity
   */
  function getSeverityClass(severity) {
    severity = severity.toLowerCase();
    if (severity.includes("serious")) return "cancer-serious";
    if (severity.includes("cancer")) return "cancer";
    if (severity.includes("pre")) return "precancer";
    return "benign";
  }

  // ========================================
  // Smooth scrolling for navigation
  // ========================================

  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute("href"));
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });
});
