// GridGuardAI — frontend logic

const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileName  = document.getElementById('file-name');
const predictBtn = document.getElementById('predict-btn');
const errorMsg  = document.getElementById('error-msg');
const loader    = document.getElementById('loader');
const uploadSec = document.getElementById('upload-section');
const resultsSec = document.getElementById('results-section');

let currentFile = null;

// ── Drag & drop ──
['dragenter', 'dragover'].forEach(ev =>
    dropZone.addEventListener(ev, e => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    })
);

['dragleave', 'drop'].forEach(ev =>
    dropZone.addEventListener(ev, e => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    })
);

dropZone.addEventListener('drop', e => {
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
});

fileInput.addEventListener('change', e => {
    if (e.target.files[0]) setFile(e.target.files[0]);
});

function setFile(f) {
    if (!f.name.endsWith('.csv')) {
        showError('Please select a CSV file');
        return;
    }
    currentFile = f;
    fileName.textContent = `✓ ${f.name} (${(f.size / 1024 / 1024).toFixed(2)} MB)`;
    predictBtn.disabled = false;
    hideError();
}

// ── Error handling ──
function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.add('visible');
}
function hideError() {
    errorMsg.classList.remove('visible');
}

// ── Predict ──
predictBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    predictBtn.disabled = true;
    hideError();
    loader.classList.remove('hidden');

    const fd = new FormData();
    fd.append('file', currentFile);

    try {
        const res = await fetch('/predict', { method: 'POST', body: fd });
        const data = await res.json();

        if (!res.ok || data.error) {
            throw new Error(data.error || 'Prediction failed');
        }

        renderResults(data);
        uploadSec.classList.add('hidden');
        resultsSec.classList.remove('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });

    } catch (err) {
        showError(err.message);
        predictBtn.disabled = false;
    } finally {
        loader.classList.add('hidden');
    }
});

// ── Render ──
function renderResults(data) {
    const s = data.summary;
    document.getElementById('total-customers').textContent = s.total_customers.toLocaleString();
    document.getElementById('flagged-count').textContent   = s.flagged.toLocaleString();
    document.getElementById('flagged-pct').textContent     = s.flagged_pct + '%';
    document.getElementById('proc-time').textContent       = s.processing_time_sec + 's';

    const tbody = document.querySelector('#results-table tbody');
    tbody.innerHTML = '';

    data.suspicious.forEach((c, i) => {
        const risk = c.risk_score;
        let cls = 'risk-low', badge = 'Low', badgeCls = 'badge-low';
        if (risk >= 0.8) { cls = 'risk-high'; badge = 'High'; badgeCls = 'badge-high'; }
        else if (risk >= 0.5) { cls = 'risk-med'; badge = 'Medium'; badgeCls = 'badge-med'; }

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${i + 1}</td>
            <td><code>${c.customer_id}</code></td>
            <td class="${cls}">${(risk * 100).toFixed(1)}%</td>
            <td><span class="badge ${badgeCls}">${badge} risk — inspect</span></td>
        `;
        tbody.appendChild(tr);
    });
}