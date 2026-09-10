const uploadForm = document.getElementById("upload-form");
const askForm = document.getElementById("ask-form");
const uploadStatus = document.getElementById("upload-status");
const askStatus = document.getElementById("ask-status");
const uploadSummary = document.getElementById("upload-summary");
const answerPanel = document.getElementById("answer-panel");
const uploadJsonPanel = document.getElementById("upload-json-panel");
const askJsonPanel = document.getElementById("ask-json-panel");
const uploadJsonInput = document.getElementById("upload-json-input");
const uploadJsonContent = document.getElementById("upload-json-content");
const askJsonInput = document.getElementById("ask-json-input");
const askJsonContent = document.getElementById("ask-json-content");
const uploadMessage = document.getElementById("upload-message");
const uploadChunks = document.getElementById("upload-chunks");
const uploadTriples = document.getElementById("upload-triples");
const answerText = document.getElementById("answer-text");
const uploadButton = document.getElementById("upload-button");
const askButton = document.getElementById("ask-button");
const fileInput = document.getElementById("pdf-file");

let uploadInProgress = false;
let uploadCompleted = false;
let uploadTimer = null;
let uploadStartedAt = 0;

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function extractErrorMessage(payload, fallbackMessage) {
  if (!payload || typeof payload !== "object") {
    return fallbackMessage;
  }
  if (typeof payload.error === "string" && payload.error.trim()) {
    return payload.error;
  }
  if (payload.detail && typeof payload.detail === "object") {
    if (typeof payload.detail.error === "string" && payload.detail.error.trim()) {
      return payload.detail.error;
    }
  }
  if (typeof payload.detail === "string" && payload.detail.trim()) {
    return payload.detail;
  }
  return fallbackMessage;
}

function setButtonsDisabled(isDisabled) {
  uploadButton.disabled = isDisabled;
  fileInput.disabled = isDisabled;
}

function setAskDisabled(isDisabled) {
  askButton.disabled = isDisabled;
}

function stopUploadTimer() {
  if (uploadTimer) {
    clearInterval(uploadTimer);
    uploadTimer = null;
  }
}

function startUploadTimer() {
  stopUploadTimer();
  uploadStartedAt = Date.now();
  uploadTimer = window.setInterval(() => {
    if (!uploadInProgress) {
      stopUploadTimer();
      return;
    }
    const elapsedSeconds = Math.max(1, Math.floor((Date.now() - uploadStartedAt) / 1000));
    uploadStatus.textContent =
      `Building the graph... ${elapsedSeconds}s elapsed. Large PDFs can take a little longer on Hebbrix free tier.`;
  }, 1000);
}

function togglePanel(panel) {
  panel.classList.toggle("hidden");
}

document.getElementById("upload-json-toggle").addEventListener("click", () => {
  togglePanel(uploadJsonPanel);
});

document.getElementById("ask-json-toggle").addEventListener("click", () => {
  togglePanel(askJsonPanel);
});

fileInput.addEventListener("change", () => {
  uploadCompleted = false;
  uploadInProgress = false;
  stopUploadTimer();
  setAskDisabled(true);
  uploadSummary.classList.add("hidden");
  answerPanel.classList.add("hidden");
  uploadStatus.textContent = fileInput.files.length
    ? "PDF selected. Build the knowledge graph to start asking questions."
    : "No PDF uploaded yet.";
  askStatus.textContent = "Ask a question after the PDF finishes processing.";
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files.length) {
    uploadStatus.textContent = "Please choose a PDF file first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  uploadJsonInput.textContent = prettyJson({
    file_name: fileInput.files[0].name,
    content_type: fileInput.files[0].type || "application/pdf",
  });

  uploadInProgress = true;
  uploadCompleted = false;
  setButtonsDisabled(true);
  setAskDisabled(true);
  uploadStatus.textContent = "Uploading the PDF and starting graph processing...";
  askStatus.textContent = "Wait for the upload to finish before asking questions.";
  uploadSummary.classList.add("hidden");
  answerPanel.classList.add("hidden");
  uploadJsonContent.textContent = prettyJson({
    status: "processing",
    message: "The PDF is being uploaded and processed. The frontend will update when the graph is ready.",
  });
  startUploadTimer();

  try {
    const response = await fetch("/upload_pdf", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    uploadJsonContent.textContent = prettyJson(data);

    if (!response.ok || data.error || (data.detail && data.detail.error)) {
      uploadStatus.textContent = extractErrorMessage(data, "The PDF could not be processed.");
      askStatus.textContent = "Upload did not finish successfully. Fix the issue and try again.";
      return;
    }

    uploadCompleted = true;
    uploadStatus.textContent =
      `Graph ready. Processed ${data.chunks_processed ?? 0} chunks and stored ${data.triples_added ?? 0} triples.`;
    uploadMessage.textContent = data.message || "PDF processed successfully.";
    uploadChunks.textContent = data.chunks_processed ?? 0;
    uploadTriples.textContent = data.triples_added ?? 0;
    uploadSummary.classList.remove("hidden");
    askStatus.textContent = "Upload complete. You can ask questions now.";
  } catch (error) {
    uploadStatus.textContent = "The upload request failed. Please try again.";
    askStatus.textContent = "Upload failed before the graph was ready.";
    uploadJsonContent.textContent = prettyJson({ error: String(error) });
  } finally {
    uploadInProgress = false;
    stopUploadTimer();
    setButtonsDisabled(false);
    setAskDisabled(!uploadCompleted);
  }
});

askForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = document.getElementById("question-input").value.trim();

  if (!question) {
    askStatus.textContent = "Please enter a question.";
    return;
  }

  if (uploadInProgress) {
    askStatus.textContent = "The PDF is still processing. Please wait until the graph is ready.";
    return;
  }

  if (!uploadCompleted) {
    askStatus.textContent = "Upload and finish processing a PDF before asking questions.";
    return;
  }

  askStatus.textContent = "Searching the knowledge graph...";
  answerPanel.classList.add("hidden");
  setAskDisabled(true);

  try {
    const requestBody = { question };
    askJsonInput.textContent = prettyJson(requestBody);
    askJsonContent.textContent = prettyJson({
      status: "processing",
      message: "The question is being searched in the knowledge graph.",
    });
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const data = await response.json();
    askJsonContent.textContent = prettyJson(data);

    if (!response.ok) {
      askStatus.textContent = extractErrorMessage(data, "The question could not be processed.");
      answerText.textContent = "";
      return;
    }

    askStatus.textContent = "Answer ready.";
    answerText.textContent =
      data.answer || "It is not in the uploaded document. Please check the text.";
    answerPanel.classList.remove("hidden");
  } catch (error) {
    askStatus.textContent = "The question request failed. Please try again.";
    askJsonContent.textContent = prettyJson({ error: String(error) });
  } finally {
    setAskDisabled(!uploadCompleted);
  }
});

setAskDisabled(true);
