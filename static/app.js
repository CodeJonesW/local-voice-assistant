const log = document.getElementById('log');
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const uploadForm = document.getElementById('upload-form');

const socket = io();
let mediaRecorder;

startBtn.onclick = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) {
            socket.emit('audio_chunk', e.data);
        }
    };
    mediaRecorder.start(250);
    startBtn.disabled = true;
    stopBtn.disabled = false;
};

stopBtn.onclick = () => {
    if (mediaRecorder) {
        mediaRecorder.stop();
        socket.emit('stop');
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
};

socket.on('transcript', data => {
    const div = document.createElement('div');
    div.textContent = 'You: ' + data.text;
    log.appendChild(div);
});

socket.on('response', data => {
    const div = document.createElement('div');
    div.textContent = 'AI: ' + data.text;
    log.appendChild(div);
});

uploadForm.onsubmit = async e => {
    e.preventDefault();
    const fileInput = document.getElementById('file');
    if (!fileInput.files.length) return;
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    await fetch('/upload', { method: 'POST', body: formData });
    fileInput.value = '';
};
