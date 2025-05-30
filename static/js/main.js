const imageInput = document.getElementById("image-input");
const previewContainer = document.getElementById("preview-container");
const previewImage = document.getElementById("preview-image");
const form = document.getElementById("predict-form");
const loading = document.getElementById("loading");
const submitButton = document.getElementById("submit-button");


imageInput.addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
        previewImage.src = e.target.result;
        previewContainer.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
    }
});


form.addEventListener("submit", function () {
    loading.classList.remove("hidden");
    submitButton.disabled = true;
    submitButton.textContent = "Submitting...";
});