const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const fileSelected = document.getElementById("fileSelected");
const fileName = document.getElementById("fileName");
const removeFile = document.getElementById("removeFile");
const summarizeBtn = document.getElementById("summarizeBtn");
const loadingIndicator = document.getElementById("loadingIndicator");
const loadingText = document.getElementById("loadingText");
const errorBox = document.getElementById("errorBox");
const errorText = document.getElementById("errorText");
const resultBox = document.getElementById("resultBox");
const summaryText = document.getElementById("summaryText");
const copyBtn = document.getElementById("copyBtn");

let selectedFile = null;

const loadingSteps = [
  "Lecture du document...",
  "Génération du résumé...",
];
let loadingStepIndex = 0;
let loadingInterval = null;

dropZone.addEventListener("click", (e) => {
  if (e.target.closest("#removeFile")) return;
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) setSelectedFile(fileInput.files[0]);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) setSelectedFile(e.dataTransfer.files[0]);
});

removeFile.addEventListener("click", (e) => {
  e.stopPropagation();
  selectedFile = null;
  fileInput.value = "";
  fileSelected.classList.add("hidden");
  summarizeBtn.disabled = true;
});

function setSelectedFile(file) {
  selectedFile = file;
  fileName.textContent = file.name;
  fileSelected.classList.remove("hidden");
  summarizeBtn.disabled = false;
  hideError();
  hideResult();
}

summarizeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  hideResult();
  startLoading();
  summarizeBtn.disabled = true;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/summarize", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
      showError(data.error || "Une erreur est survenue.");
    } else {
      showResult(data);
    }
  } catch (err) {
    showError("Impossible de contacter le serveur. Vérifie que l'application Flask est bien lancée.");
  } finally {
    stopLoading();
    summarizeBtn.disabled = false;
  }
});

copyBtn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(summaryText.textContent);
  copyBtn.classList.add("copied");
  copyBtn.querySelector("span")?.remove();
  const label = copyBtn.childNodes[copyBtn.childNodes.length - 1];
  const original = label.textContent;
  label.textContent = " Copié !";
  setTimeout(() => {
    label.textContent = original;
    copyBtn.classList.remove("copied");
  }, 1800);
});

function startLoading() {
  loadingStepIndex = 0;
  loadingText.textContent = loadingSteps[0];
  loadingIndicator.classList.remove("hidden");
  loadingInterval = setInterval(() => {
    loadingStepIndex = (loadingStepIndex + 1) % loadingSteps.length;
    loadingText.textContent = loadingSteps[loadingStepIndex];
  }, 1400);
}

function stopLoading() {
  clearInterval(loadingInterval);
  loadingIndicator.classList.add("hidden");
}

function showResult(data) {
  summaryText.textContent = data.summary;
  resultBox.classList.remove("hidden");
  resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function hideResult() {
  resultBox.classList.add("hidden");
}

function showError(message) {
  errorText.textContent = message;
  errorBox.classList.remove("hidden");
}

function hideError() {
  errorBox.classList.add("hidden");
}