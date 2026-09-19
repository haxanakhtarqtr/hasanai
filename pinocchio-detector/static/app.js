const form = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');
const progressDiv = document.getElementById('progress');
const outputPre = document.getElementById('output');
const downloadLink = document.getElementById('downloadLink');
let pollInterval = null;

form.addEventListener('submit', function(e) {
  e.preventDefault();
  const formData = new FormData(form);
  const file = formData.get('video');
  if (!file) return;

  progressDiv.textContent = 'Uploading and starting analysis…';
  resultDiv.style.display = 'block';
  outputPre.textContent = '';
  downloadLink.style.display = 'none';
  form.querySelector('button').disabled = true;

  fetch('/upload', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      const jobId = data.job_id;
      pollInterval = setInterval(() => pollStatus(jobId), 1000);
    })
    .catch(err => {
      progressDiv.textContent = 'Upload failed: ' + err.message;
      form.querySelector('button').disabled = false;
    });
});

function pollStatus(jobId) {
  fetch(`/status/${jobId}`)
    .then(r => r.json())
    .then(data => {
      progressDiv.textContent = `${data.progress}% — ${data.message}`;
      if (data.status === 'completed') {
        clearInterval(pollInterval);
        form.querySelector('button').disabled = false;
        fetchResult(jobId);
      } else if (data.status === 'error') {
        clearInterval(pollInterval);
        progressDiv.textContent = 'Error: ' + (data.error || data.message);
        form.querySelector('button').disabled = false;
      }
    });
}

function fetchResult(jobId) {
  fetch(`/result/${jobId}`)
    .then(r => r.json())
    .then(data => {
      outputPre.textContent = JSON.stringify(data, null, 2);
      downloadLink.href = `/report/${jobId}`;
      downloadLink.style.display = 'inline-block';
      progressDiv.textContent = data.verdict === 'GENUINE'
        ? `✅ Genuine (score: ${data.score})`
        : `🚨 Deepfake detected (score: ${data.score})`;
    });
}