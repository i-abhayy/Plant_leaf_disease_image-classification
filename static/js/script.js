
document.addEventListener("DOMContentLoaded", () => {

    // ==========================================
    // 1. GET HTML ELEMENTS
    // ==========================================

    const imageInput = document.getElementById("imageInput");
    const uploadArea = document.getElementById("uploadArea");
    const browseButton = document.getElementById("browseButton");
    const uploadContent = document.getElementById("uploadContent");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const removeImage = document.getElementById("removeImage");
    const analyzeButton = document.getElementById("analyzeButton");
    const statusMessage = document.getElementById("statusMessage");

    const resultSection = document.getElementById("resultSection");
    const predictionStatus = document.getElementById("predictionStatus");
    const diseaseName = document.getElementById("diseaseName");
    const diseaseDescription = document.getElementById("diseaseDescription");
    const confidence = document.getElementById("confidence");
    const confidenceBar = document.getElementById("confidenceBar");
    const diseaseCauses = document.getElementById("diseaseCauses");
    const diseaseSolutions = document.getElementById("diseaseSolutions");
    const scanAgainButton = document.getElementById("scanAgainButton");
    const geminiDetails = document.getElementById("geminiDetails");

    const requiredElements = [
        imageInput,
        uploadArea,
        browseButton,
        imagePreview,
        analyzeButton,
        statusMessage,
        resultSection,
        geminiDetails
    ];

    if (requiredElements.some(element => !element)) {
        console.error("Some required HTML elements are missing.");
        return;
    }


    // ==========================================
    // 2. VARIABLES
    // ==========================================

    let selectedFile = null;
    let currentImageURL = null;

    const MAX_FILE_SIZE = 8 * 1024 * 1024;


    // ==========================================
    // 3. HELPER FUNCTIONS
    // ==========================================

    function updateStatus(message) {
        statusMessage.textContent = message;
    }


    function formatClassName(value) {
        return String(value || "Unknown")
            .replace(/___/g, " - ")
            .replace(/_/g, " ");
    }


    function normalizeList(value) {
        if (Array.isArray(value)) {
            return value;
        }

        if (typeof value === "string" && value.trim()) {
            return [value];
        }

        return [];
    }


    function showList(element, items) {
        if (!element) return;

        const listItems = normalizeList(items);

        element.replaceChildren();

        if (listItems.length === 0) {
            element.textContent = "No information available.";
            return;
        }

        const list = document.createElement("ul");

        listItems.forEach(item => {
            const li = document.createElement("li");
            li.textContent = String(item);
            list.appendChild(li);
        });

        element.appendChild(list);
    }


    // ==========================================
    // 4. CREATE GEMINI INSIGHT CARDS
    // ==========================================

    function createInsightCard(title, content, icon) {

        const card = document.createElement("div");
        card.className = "detail-card gemini-detail-card";

        const iconElement = document.createElement("span");
        iconElement.className = "detail-icon";
        iconElement.textContent = icon;

        const heading = document.createElement("h4");
        heading.textContent = title;

        const contentContainer = document.createElement("div");
        contentContainer.className = "insight-content";

        if (Array.isArray(content)) {

            const items = normalizeList(content);

            if (items.length > 0) {

                const list = document.createElement("ul");

                items.forEach(item => {
                    const li = document.createElement("li");
                    li.textContent = String(item);
                    list.appendChild(li);
                });

                contentContainer.appendChild(list);

            } else {

                const paragraph = document.createElement("p");
                paragraph.textContent = "No additional information available.";
                contentContainer.appendChild(paragraph);

            }

        } else {

            const paragraph = document.createElement("p");
            paragraph.textContent =
                content || "No additional information available.";

            contentContainer.appendChild(paragraph);
        }

        card.appendChild(iconElement);
        card.appendChild(heading);
        card.appendChild(contentContainer);

        return card;
    }


    // ==========================================
    // 5. IMAGE PREVIEW
    // ==========================================

    function displayImage(file) {

        if (!file) return;

        if (!file.type.startsWith("image/")) {
            updateStatus("Please select a valid image.");
            return;
        }

        if (file.size > MAX_FILE_SIZE) {
            updateStatus("Image must be smaller than 8 MB.");
            return;
        }

        selectedFile = file;

        if (currentImageURL) {
            URL.revokeObjectURL(currentImageURL);
        }

        currentImageURL = URL.createObjectURL(file);

        imagePreview.src = currentImageURL;
        imagePreview.hidden = false;

        if (uploadContent) {
            uploadContent.hidden = true;
        }

        if (previewContainer) {
            previewContainer.hidden = false;
        }

        analyzeButton.disabled = false;

        resultSection.hidden = true;

        updateStatus("Image uploaded successfully. Ready to analyze.");
    }


    function resetImage() {

        selectedFile = null;
        imageInput.value = "";

        if (currentImageURL) {
            URL.revokeObjectURL(currentImageURL);
            currentImageURL = null;
        }

        imagePreview.src = "";
        imagePreview.hidden = true;

        if (previewContainer) {
            previewContainer.hidden = true;
        }

        if (uploadContent) {
            uploadContent.hidden = false;
        }

        analyzeButton.disabled = true;

        resultSection.hidden = true;

        geminiDetails.replaceChildren();

        updateStatus("Upload a leaf image to begin.");
    }


    // Browse button
    browseButton.addEventListener("click", event => {
        event.stopPropagation();
        imageInput.click();
    });


    // Upload area click
    uploadArea.addEventListener("click", event => {

        if (
            event.target === browseButton ||
            event.target.closest("#browseButton") ||
            event.target.closest("#removeImage")
        ) {
            return;
        }

        imageInput.click();
    });


    // File selection
    imageInput.addEventListener("change", () => {

        const file = imageInput.files[0];

        if (file) {
            displayImage(file);
        }
    });


    // Remove image
    if (removeImage) {

        removeImage.addEventListener("click", event => {
            event.stopPropagation();
            resetImage();
        });
    }


    // ==========================================
    // 6. DRAG AND DROP
    // ==========================================

    ["dragenter", "dragover"].forEach(eventName => {

        uploadArea.addEventListener(eventName, event => {

            event.preventDefault();
            uploadArea.classList.add("drag-active");

        });
    });


    ["dragleave", "drop"].forEach(eventName => {

        uploadArea.addEventListener(eventName, event => {

            event.preventDefault();
            uploadArea.classList.remove("drag-active");

        });
    });


    uploadArea.addEventListener("drop", event => {

        const file = event.dataTransfer.files[0];

        if (file) {
            displayImage(file);
        }
    });


    // ==========================================
    // 7. DISPLAY FINAL RESULTS
    // ==========================================

    function displayResults(data) {

        const model = data.model_prediction || {};
        const gemini = data.gemini_assessment || {};
        const finalResult = data.final_result || {};


        // --------------------------------------
        // MAIN DIAGNOSIS
        // --------------------------------------

        if (diseaseName) {

            diseaseName.textContent =
                finalResult.disease ||
                gemini.final_condition ||
                "Unable to determine";
        }


        if (diseaseDescription) {

            diseaseDescription.textContent =
                finalResult.explanation ||
                gemini.explanation ||
                "No explanation available.";
        }


        // --------------------------------------
        // MOBILENETV2 CONFIDENCE
        // --------------------------------------

        const modelConfidence = Number(model.confidence);

        if (confidence) {

            confidence.textContent =
                Number.isFinite(modelConfidence)
                    ? `${modelConfidence.toFixed(2)}%`
                    : "Unavailable";
        }


        if (confidenceBar) {

            const safeConfidence = Number.isFinite(modelConfidence)
                ? Math.min(100, Math.max(0, modelConfidence))
                : 0;

            confidenceBar.style.width = `${safeConfidence}%`;
        }


        // --------------------------------------
        // GEMINI VERIFICATION STATUS
        // --------------------------------------

        const verificationStatus =
            finalResult.status || "uncertain";

        const statusMessages = {

            agree: "✓ Gemini and MobileNetV2 assessments agree.",

            disagree: "⚠ Gemini and MobileNetV2 assessments differ.",

            uncertain: "Diagnosis uncertain. Further inspection is needed."
        };


        if (predictionStatus) {

            predictionStatus.textContent =
                statusMessages[verificationStatus] ||
                statusMessages.uncertain;

            predictionStatus.className =
                `result-status ${verificationStatus}`;
        }


        // --------------------------------------
        // POSSIBLE CAUSES
        // --------------------------------------

        showList(
            diseaseCauses,
            gemini.possible_causes || model.causes || []
        );


        // --------------------------------------
        // SUGGESTED SOLUTIONS
        // --------------------------------------

        const geminiSolutions = [

            ...normalizeList(gemini.immediate_actions),

            ...normalizeList(gemini.care_recommendations)
        ];

        showList(
            diseaseSolutions,
            geminiSolutions.length > 0
                ? geminiSolutions
                : (model.solutions || [])
        );


        // --------------------------------------
        // ADDITIONAL GEMINI INSIGHTS
        // --------------------------------------

        geminiDetails.replaceChildren();


        // 1. Original MobileNetV2 prediction

        geminiDetails.appendChild(
            createInsightCard(

                "MobileNetV2 Original Prediction",

                `${formatClassName(model.disease)} | Confidence: ${
                    Number.isFinite(modelConfidence)
                        ? modelConfidence.toFixed(2) + "%"
                        : "Unavailable"
                }`,

                "🔬"
            )
        );


        // 2. Plant identification

        geminiDetails.appendChild(
            createInsightCard(

                "Gemini Plant Identification",

                finalResult.plant ||
                gemini.plant_name ||
                "Unable to identify the plant.",

                "🌱"
            )
        );


        // 3. Visual observations

        geminiDetails.appendChild(
            createInsightCard(

                "Visual Observations",

                gemini.visual_observations || [],

                "👁️"
            )
        );


        // 4. Symptoms to check

        geminiDetails.appendChild(
            createInsightCard(

                "Symptoms to Check",

                gemini.symptoms_to_check || [],

                "🔎"
            )
        );


        // 5. Prevention tips

        geminiDetails.appendChild(
            createInsightCard(

                "Prevention Tips",

                gemini.prevention_tips || [],

                "🌿"
            )
        );


        // 6. Important caution

        geminiDetails.appendChild(
            createInsightCard(

                "Important Caution",

                finalResult.caution ||
                gemini.caution ||
                "Image-based analysis cannot replace professional verification.",

                "⚠️"
            )
        );


        // --------------------------------------
        // SHOW RESULTS
        // --------------------------------------

        resultSection.hidden = false;

        resultSection.scrollIntoView({

            behavior: "smooth",

            block: "start"
        });
    }


    // ==========================================
    // 8. ANALYZE IMAGE USING FLASK /VERIFY
    // ==========================================

    analyzeButton.addEventListener("click", async () => {

        if (!selectedFile) {

            updateStatus("Please upload a leaf image first.");
            return;
        }


        const formData = new FormData();

        formData.append("file", selectedFile);


        analyzeButton.disabled = true;
        analyzeButton.textContent = "Analyzing...";

        updateStatus(
            "MobileNetV2 is analyzing your image and Gemini is verifying it..."
        );


        try {

            const response = await fetch("/verify", {

                method: "POST",

                body: formData
            });


            const data = await response.json();


            if (!response.ok || !data.success) {

                throw new Error(
                    data.error || "Unable to analyze this image."
                );
            }


            displayResults(data);

            updateStatus(
                "Analysis completed successfully. Review your plant health report."
            );


        } catch (error) {

            console.error("LeafCare AI error:", error);

            updateStatus(
                error.message || "Something went wrong during analysis."
            );


        } finally {

            analyzeButton.disabled = !selectedFile;

            analyzeButton.textContent = "Analyze My Plant";
        }

    });


    // ==========================================
    // 9. SCAN ANOTHER LEAF
    // ==========================================

    if (scanAgainButton) {

        scanAgainButton.addEventListener("click", () => {

            resetImage();

            window.scrollTo({

                top: 0,

                behavior: "smooth"
            });

        });
    }

});
