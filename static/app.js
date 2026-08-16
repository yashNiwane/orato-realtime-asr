/**
 * Orato Realtime Hindi ASR Web Client
 * Delivers immediate live word-by-word transcription as speech occurs.
 */

// DOM Elements
const connectionStatus = document.getElementById("connectionStatus");
const statusText = connectionStatus.querySelector(".status-text");
const languageSelect = document.getElementById("languageSelect");
const micToggleBtn = document.getElementById("micToggleBtn");
const recordBtnText = document.getElementById("recordBtnText");
const flushBtn = document.getElementById("flushBtn");
const copyBtn = document.getElementById("copyBtn");
const clearBtn = document.getElementById("clearBtn");
const vadBadge = document.getElementById("vadBadge");
const vuBar = document.getElementById("vuBar");
const interimText = document.getElementById("interimText");
const transcriptTimeline = document.getElementById("transcriptTimeline");
const emptyState = document.getElementById("emptyState");
const utteranceCount = document.getElementById("utteranceCount");
const statLatency = document.getElementById("statLatency");
const statBackend = document.getElementById("statBackend");
const dropZone = document.getElementById("dropZone");
const audioFileInput = document.getElementById("audioFileInput");
const uploadProgressWrap = document.getElementById("uploadProgressWrap");
const waveformCanvas = document.getElementById("waveformCanvas");
const canvasCtx = waveformCanvas.getContext("2d");
const customEndpointInput = document.getElementById("customEndpointInput");
const connectEndpointBtn = document.getElementById("connectEndpointBtn");

// State
let ws = null;
let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;
let isRecording = false;
let confirmedUtterances = [];
let analyser = null;
let animFrameId = null;
let activeBaseUrl = localStorage.getItem("orato_backend_url") || "";

if (activeBaseUrl && customEndpointInput) {
    customEndpointInput.value = activeBaseUrl;
}

// Compute WebSocket URL
function getWsUrl() {
    const selectedLang = languageSelect.value;
    if (activeBaseUrl) {
        let clean = activeBaseUrl.trim().replace(/\/+$/, "");
        if (clean.startsWith("http://")) {
            clean = "ws://" + clean.substring(7);
        } else if (clean.startsWith("https://")) {
            clean = "wss://" + clean.substring(8);
        } else if (!clean.startsWith("ws://") && !clean.startsWith("wss://")) {
            clean = "wss://" + clean;
        }
        return `${clean}/ws/transcribe?language=${encodeURIComponent(selectedLang)}`;
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws/transcribe?language=${encodeURIComponent(selectedLang)}`;
}

function getHttpBaseUrl() {
    if (activeBaseUrl) {
        let clean = activeBaseUrl.trim().replace(/\/+$/, "");
        if (clean.startsWith("ws://")) {
            clean = "http://" + clean.substring(5);
        } else if (clean.startsWith("wss://")) {
            clean = "https://" + clean.substring(6);
        } else if (!clean.startsWith("http://") && !clean.startsWith("https://")) {
            clean = "https://" + clean;
        }
        return clean;
    }
    return window.location.origin;
}

// Init WebSocket connection
function initWebSocket() {
    if (ws) {
        try { ws.close(); } catch(e){}
    }

    const wsUrl = getWsUrl();
    connectionStatus.className = "status-chip";
    statusText.textContent = "Connecting...";

    try {
        ws = new WebSocket(wsUrl);
        ws.binaryType = "arraybuffer";

        ws.onopen = () => {
            connectionStatus.className = "status-chip connected";
            statusText.textContent = activeBaseUrl ? "Connected (Remote GPU)" : "Connected (Local)";
            updateBackendInfo();
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleServerMessage(data);
            } catch (e) {
                console.error("Error parsing WS message:", e);
            }
        };

        ws.onclose = () => {
            connectionStatus.className = "status-chip disconnected";
            statusText.textContent = "Disconnected (Retrying...)";
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
            connectionStatus.className = "status-chip disconnected";
            statusText.textContent = "Connection Failed";
        };
    } catch (err) {
        console.error("WS Init error:", err);
    }
}

function handleServerMessage(data) {
    if (data.type === "connected") {
        statBackend.textContent = (data.model || "Orato ASR").split("/").pop().split("\\").pop();
    } else if (data.type === "speech_start") {
        vadBadge.className = "vad-badge speaking";
        vadBadge.textContent = "Speaking";
    } else if (data.type === "speech_end") {
        vadBadge.className = "vad-badge";
        vadBadge.textContent = "Idle";
    } else if (data.type === "partial") {
        if (data.text) {
            interimText.innerHTML = `<span class="live-word">${escapeHtml(data.text)}</span> <span class="typewriter-cursor"></span>`;
        }
        if (data.latency_ms) {
            statLatency.textContent = `${data.latency_ms} ms`;
        }
    } else if (data.type === "final") {
        interimText.innerHTML = `Listening for Hindi / Hinglish speech...`;
        if (data.text) {
            addConfirmedUtterance(data);
        }
        if (data.latency_ms) {
            statLatency.textContent = `${data.latency_ms} ms`;
        }
    }
}

function addConfirmedUtterance(data) {
    confirmedUtterances.push(data);
    updateUtteranceCount();

    if (emptyState) {
        emptyState.style.display = "none";
    }

    const item = document.createElement("div");
    item.className = "utterance-item";
    
    const timeStr = new Date().toLocaleTimeString();
    const duration = data.duration_sec ? `${data.duration_sec}s` : "";
    const latency = data.latency_ms ? `⚡ ${data.latency_ms}ms` : "";

    item.innerHTML = `
        <div class="utterance-meta">
            <span class="utterance-lang-badge">${data.language || "Hindi"}</span>
            <span>${timeStr} ${duration ? "• " + duration : ""} ${latency ? "• " + latency : ""}</span>
        </div>
        <div class="utterance-body devanagari-font">${escapeHtml(data.text)}</div>
    `;

    transcriptTimeline.appendChild(item);
    transcriptTimeline.scrollTop = transcriptTimeline.scrollHeight;
}

function updateUtteranceCount() {
    utteranceCount.textContent = `${confirmedUtterances.length} utterance${confirmedUtterances.length === 1 ? "" : "s"}`;
}

// Start Microphone Capture
async function startRecording() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });

        const source = audioContext.createMediaStreamSource(mediaStream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);

        // 2048 buffer size gives ~128ms per chunk at 16kHz for instant reactivity
        scriptProcessor = audioContext.createScriptProcessor(2048, 1, 1);
        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        scriptProcessor.onaudioprocess = (e) => {
            if (!isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;

            const inputData = e.inputBuffer.getChannelData(0);
            
            // RMS calculation for VU meter
            let sum = 0;
            for (let i = 0; i < inputData.length; i++) {
                sum += inputData[i] * inputData[i];
            }
            const rms = Math.sqrt(sum / inputData.length);
            const level = Math.min(100, Math.round(rms * 500));
            vuBar.style.width = `${level}%`;

            // Convert to 16-bit PCM binary
            const pcm16 = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                const s = Math.max(-1, Math.min(1, inputData[i]));
                pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }

            ws.send(pcm16.buffer);
        };

        isRecording = true;
        micToggleBtn.className = "btn btn-primary btn-record recording";
        recordBtnText.textContent = "Stop Recording";
        flushBtn.disabled = false;
        interimText.innerHTML = `Listening... <span class="typewriter-cursor"></span>`;
        
        drawWaveform();
    } catch (err) {
        console.error("Microphone access error:", err);
        alert("Could not access microphone: " + err.message);
    }
}

// Stop Microphone Capture
function stopRecording() {
    isRecording = false;
    micToggleBtn.className = "btn btn-primary btn-record";
    recordBtnText.textContent = "Start Realtime Mic";
    flushBtn.disabled = true;
    vuBar.style.width = "0%";
    vadBadge.className = "vad-badge";
    vadBadge.textContent = "Idle";

    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    if (audioContext && audioContext.state !== "closed") {
        audioContext.close();
        audioContext = null;
    }
    if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "flush" }));
    }
}

// Waveform Visualizer Loop
function drawWaveform() {
    if (!isRecording || !analyser) {
        canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        return;
    }

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = "#090d16";
    canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = "#06b6d4";
    canvasCtx.beginPath();

    const sliceWidth = waveformCanvas.width * 1.0 / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * waveformCanvas.height / 2;

        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }

        x += sliceWidth;
    }

    canvasCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
    canvasCtx.stroke();

    animFrameId = requestAnimationFrame(drawWaveform);
}

// File Upload & Transcribe
async function handleFileUpload(file) {
    if (!file) return;

    uploadProgressWrap.style.display = "flex";
    const formData = new FormData();
    formData.append("file", file);
    formData.append("language", languageSelect.value);

    try {
        const startTime = performance.now();
        const base = getHttpBaseUrl();
        const response = await fetch(`${base}/api/v1/transcribe`, {
            method: "POST",
            body: formData
        });
        const elapsed = Math.round(performance.now() - startTime);

        const data = await response.json();
        uploadProgressWrap.style.display = "none";

        if (data.success && data.text) {
            addConfirmedUtterance({
                text: data.text,
                language: data.language,
                duration_sec: data.duration_sec,
                latency_ms: data.latency_ms || elapsed
            });
        } else {
            alert("Transcription completed: No speech recognized in file.");
        }
    } catch (err) {
        uploadProgressWrap.style.display = "none";
        console.error("Upload error:", err);
        alert("Transcription failed: " + err.message);
    }
}

// Event Listeners
micToggleBtn.addEventListener("click", () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

flushBtn.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "flush" }));
    }
});

languageSelect.addEventListener("change", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            action: "set_language",
            language: languageSelect.value
        }));
    }
});

copyBtn.addEventListener("click", () => {
    const fullText = confirmedUtterances.map(u => u.text).join("\n");
    if (fullText) {
        navigator.clipboard.writeText(fullText).then(() => {
            copyBtn.textContent = "Copied!";
            setTimeout(() => {
                copyBtn.innerHTML = `
                    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg> Copy
                `;
            }, 1500);
        });
    }
});

clearBtn.addEventListener("click", () => {
    confirmedUtterances = [];
    updateUtteranceCount();
    transcriptTimeline.innerHTML = "";
    if (emptyState) {
        transcriptTimeline.appendChild(emptyState);
        emptyState.style.display = "flex";
    }
    interimText.textContent = "Listening for Hindi / Hinglish speech...";
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "reset" }));
    }
});

// Custom Endpoint Connect Button
if (connectEndpointBtn && customEndpointInput) {
    connectEndpointBtn.addEventListener("click", () => {
        const val = customEndpointInput.value.trim();
        activeBaseUrl = val;
        localStorage.setItem("orato_backend_url", val);
        initWebSocket();
    });
    customEndpointInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            connectEndpointBtn.click();
        }
    });
}

// Drag & Drop
dropZone.addEventListener("click", () => audioFileInput.click());
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});
audioFileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function updateBackendInfo() {
    const base = getHttpBaseUrl();
    fetch(`${base}/health`)
        .then(r => r.json())
        .then(data => {
            if (data.model) {
                statBackend.textContent = (data.model || "").split("/").pop().split("\\").pop() + (data.device ? ` (${data.device.toUpperCase()})` : "");
            }
        })
        .catch(() => {});
}

// Initialize on page load
window.addEventListener("DOMContentLoaded", () => {
    initWebSocket();
});
