/**
 * Vision Search Agent — Frontend Logic (Tabbed UI)
 */

const API = '';  // Same origin

// ---- DOM Elements ----
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const searchMode = document.getElementById('searchMode');
const toggleIndex = document.getElementById('toggleIndex');
const indexPanel = document.getElementById('indexPanel');
const indexPath = document.getElementById('indexPath');
const indexBtn = document.getElementById('indexBtn');
const useClip = document.getElementById('useClip');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultsSection = document.getElementById('resultsSection');
const resultsTitle = document.getElementById('resultsTitle');
const resultsCount = document.getElementById('resultsCount');
const resultsGrid = document.getElementById('resultsGrid');
const videosList = document.getElementById('videosList');
const videoCount = document.getElementById('videoCount');
const modalOverlay = document.getElementById('modalOverlay');
const modalImage = document.getElementById('modalImage');
const modalInfo = document.getElementById('modalInfo');
const modalClose = document.getElementById('modalClose');

// ---- State ----
let pollTimer = null;
let activeTab = 'text';
let pendingImageFile = null;
let pendingPersonFile = null;

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    loadVideos();

    // ---- Tab Switching ----
    document.querySelectorAll('.search-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;
            switchTab(targetTab);
        });
    });

    // ---- Text Search ----
    searchBtn.addEventListener('click', doTextSearch);
    searchInput.addEventListener('keydown', e => {
        if (e.key === 'Enter') doTextSearch();
    });

    // ---- Visual Search (Image tab) ----
    const imageDropZone = document.getElementById('imageDropZone');
    const imageFileInput = document.getElementById('imageFileInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const clearImage = document.getElementById('clearImage');
    const imageSearchBtn = document.getElementById('imageSearchBtn');

    imageDropZone.addEventListener('click', () => imageFileInput.click());
    imageDropZone.addEventListener('dragover', e => { e.preventDefault(); imageDropZone.classList.add('drag-over'); });
    imageDropZone.addEventListener('dragleave', () => imageDropZone.classList.remove('drag-over'));
    imageDropZone.addEventListener('drop', e => {
        e.preventDefault();
        imageDropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) setImageFile(file);
    });
    imageFileInput.addEventListener('change', e => {
        if (e.target.files[0]) setImageFile(e.target.files[0]);
    });
    clearImage.addEventListener('click', () => {
        pendingImageFile = null;
        imagePreview.style.display = 'none';
        imageDropZone.style.display = '';
        imageFileInput.value = '';
    });
    imageSearchBtn.addEventListener('click', doImageSearch);

    function setImageFile(file) {
        pendingImageFile = file;
        previewImg.src = URL.createObjectURL(file);
        imagePreview.style.display = 'flex';
        imageDropZone.style.display = 'none';
    }

    // ---- Find Person (Person tab) ----
    const personDropZone = document.getElementById('personDropZone');
    const personFileInput = document.getElementById('personFileInput');
    const personPreview = document.getElementById('personPreview');
    const personPreviewImg = document.getElementById('personPreviewImg');
    const clearPerson = document.getElementById('clearPerson');
    const personSearchBtn = document.getElementById('personSearchBtn');

    personDropZone.addEventListener('click', () => personFileInput.click());
    personDropZone.addEventListener('dragover', e => { e.preventDefault(); personDropZone.classList.add('drag-over'); });
    personDropZone.addEventListener('dragleave', () => personDropZone.classList.remove('drag-over'));
    personDropZone.addEventListener('drop', e => {
        e.preventDefault();
        personDropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) setPersonFile(file);
    });
    personFileInput.addEventListener('change', e => {
        if (e.target.files[0]) setPersonFile(e.target.files[0]);
    });
    clearPerson.addEventListener('click', () => {
        pendingPersonFile = null;
        personPreview.style.display = 'none';
        personDropZone.style.display = '';
        personFileInput.value = '';
    });
    personSearchBtn.addEventListener('click', doPersonSearch);

    function setPersonFile(file) {
        pendingPersonFile = file;
        personPreviewImg.src = URL.createObjectURL(file);
        personPreview.style.display = 'flex';
        personDropZone.style.display = 'none';
    }

    // ---- Index panel toggle ----
    toggleIndex.addEventListener('click', () => {
        toggleIndex.classList.toggle('active');
        indexPanel.classList.toggle('open');
    });
    indexBtn.addEventListener('click', startIndexing);

    // ---- Modal ----
    modalClose.addEventListener('click', closeModal);
    modalOverlay.addEventListener('click', e => {
        if (e.target === modalOverlay) closeModal();
    });
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeModal();
    });
});

// ---- Tab Switching ----
function switchTab(tab) {
    activeTab = tab;
    document.querySelectorAll('.search-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');
}

// ---- Text Search ----
async function doTextSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    searchBtn.innerHTML = '<span class="spinner"></span>';
    try {
        const mode = searchMode.value;
        const resp = await fetch(`${API}/api/search?q=${encodeURIComponent(query)}&top_k=30&mode=${mode}`);
        const data = await resp.json();
        showResults(`Results for "${query}"`, data.count, data.results, false);
    } catch (err) {
        console.error('Search failed:', err);
    } finally {
        searchBtn.textContent = 'Search';
    }
}

// ---- Visual Image Search ----
async function doImageSearch() {
    if (!pendingImageFile) return;
    const btn = document.getElementById('imageSearchBtn');
    btn.innerHTML = '<span class="spinner"></span>';
    try {
        const formData = new FormData();
        formData.append('file', pendingImageFile);
        const resp = await fetch(`${API}/api/search-by-image?top_k=30`, { method: 'POST', body: formData });
        const data = await resp.json();
        showResults(`Visual match: "${pendingImageFile.name}"`, data.count, data.results, false);
    } catch (err) {
        console.error('Image search failed:', err);
    } finally {
        btn.textContent = 'Search';
    }
}

// ---- Person Re-ID Search ----
async function doPersonSearch() {
    if (!pendingPersonFile) return;
    const btn = document.getElementById('personSearchBtn');
    btn.innerHTML = '<span class="spinner"></span>';
    try {
        const formData = new FormData();
        formData.append('file', pendingPersonFile);
        const resp = await fetch(`${API}/api/search-by-person?top_k=30`, { method: 'POST', body: formData });
        const data = await resp.json();
        showResults(`Person search: "${pendingPersonFile.name}"`, data.count, data.results, true);
    } catch (err) {
        console.error('Person search failed:', err);
    } finally {
        btn.textContent = 'Find This Person';
    }
}

// ---- Show Results ----
function showResults(title, count, results, isPersonSearch) {
    resultsSection.style.display = 'block';
    resultsTitle.textContent = title;
    resultsCount.textContent = `${count} found`;
    renderResults(results, isPersonSearch);
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderResults(results, isPersonSearch = false) {
    if (!results || results.length === 0) {
        resultsGrid.innerHTML = '<p class="empty-state">No results found. Try a different query or index more videos.</p>';
        return;
    }

    window._searchResults = results;

    resultsGrid.innerHTML = results.map((r, i) => {
        const imgSrc = r.crop_path
            ? `${API}/api/frame?path=${encodeURIComponent(r.crop_path)}`
            : `${API}/api/frame?path=${encodeURIComponent(r.frame_path)}`;

        const frameSrc = `${API}/api/frame?path=${encodeURIComponent(r.frame_path)}`;
        const time = formatTime(r.timestamp);
        const scorePercent = Math.min(100, Math.round((r.score || 0) * 100));

        // Match type badge for person search
        let matchBadge = '';
        if (isPersonSearch && r.match_type) {
            const colors = { 'face': '#22c55e', 'body': '#3b82f6', 'body+face': '#f59e0b' };
            const labels = { 'face': '👤 Face', 'body': '🚶 Body', 'body+face': '✨ Face+Body' };
            matchBadge = `<span class="match-badge" style="background:${colors[r.match_type] || '#6366f1'}">${labels[r.match_type] || r.match_type}</span>`;
        }

        // Source badge styling
        const sourceBadge = isPersonSearch
            ? matchBadge
            : `<span class="source-badge">${r.source}</span>`;

        return `
            <div class="result-card" data-index="${i}" data-frame-src="${frameSrc}">
                <div class="image-container">
                    <img src="${imgSrc}" alt="Result ${i + 1}" loading="lazy">
                    <div class="badge-row">
                        <span class="type-badge ${r.type}">${r.type}</span>
                        ${sourceBadge}
                    </div>
                    <span class="timestamp-badge">${time}</span>
                    ${scorePercent > 0 ? `<span class="score-badge">${scorePercent}%</span>` : ''}
                </div>
                <div class="result-info">
                    ${r.caption && r.caption !== r.class_name ? `<p class="caption">${escapeHtml(r.caption)}</p>` : ''}
                    <div class="meta">
                        <span>${r.video_filename || 'Unknown'}</span>
                        <div class="score-bar">
                            <div class="score-bar-fill" style="width: ${scorePercent}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Event delegation for card clicks
    resultsGrid.querySelectorAll('.result-card').forEach(card => {
        card.addEventListener('click', () => {
            const idx = parseInt(card.dataset.index);
            const r = window._searchResults[idx];
            const frameSrc = card.dataset.frameSrc;
            openModal(frameSrc, r);
        });
    });
}

// ---- Indexing ----
async function startIndexing() {
    const path = indexPath.value.trim();
    if (!path) {
        indexPath.style.borderColor = 'var(--danger)';
        setTimeout(() => indexPath.style.borderColor = '', 1500);
        return;
    }

    indexBtn.disabled = true;
    indexBtn.textContent = 'Starting...';

    try {
        const resp = await fetch(`${API}/api/index`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                path: path,
                use_vlm: false,
                use_clip: useClip.checked,
            })
        });

        const data = await resp.json();

        if (!resp.ok) {
            alert(data.error || 'Indexing failed');
            indexBtn.disabled = false;
            indexBtn.textContent = 'Start Indexing';
            return;
        }

        progressSection.style.display = 'block';
        startPolling();
    } catch (err) {
        console.error('Index error:', err);
        indexBtn.disabled = false;
        indexBtn.textContent = 'Start Indexing';
    }
}

function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
        try {
            const resp = await fetch(`${API}/api/status`);
            const status = await resp.json();

            progressFill.style.width = `${status.progress}%`;
            progressText.textContent = status.message || 'Processing...';

            if (!status.active && status.progress >= 100) {
                clearInterval(pollTimer);
                pollTimer = null;
                indexBtn.disabled = false;
                indexBtn.textContent = 'Start Indexing';
                loadVideos();

                setTimeout(() => {
                    progressSection.style.display = 'none';
                    progressFill.style.width = '0%';
                }, 3000);
            }
        } catch (err) {
            console.error('Status poll error:', err);
        }
    }, 1000);
}

// ---- Videos List ----
async function loadVideos() {
    try {
        const resp = await fetch(`${API}/api/videos`);
        const data = await resp.json();
        const videos = data.videos || [];

        videoCount.textContent = `${videos.length} video${videos.length !== 1 ? 's' : ''}`;

        if (videos.length === 0) {
            videosList.innerHTML = '<p class="empty-state">No videos indexed yet. Add a video path above to get started.</p>';
            return;
        }

        videosList.innerHTML = videos.map(v => `
            <div class="video-item">
                <div class="video-info">
                    <div class="video-icon">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polygon points="23 7 16 12 23 17 23 7"/>
                            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
                        </svg>
                    </div>
                    <div>
                        <div class="video-name">${escapeHtml(v.filename)}</div>
                        <div class="video-meta">${v.duration ? formatTime(v.duration) : 'N/A'} · ${v.total_frames || 0} frames</div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <span class="video-status ${v.status}">${v.status}</span>
                    <button class="delete-btn" onclick="deleteVideo(${v.id})" title="Delete">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                        </svg>
                    </button>
                </div>
            </div>
        `).join('');
    } catch (err) {
        console.error('Load videos error:', err);
    }
}

async function deleteVideo(id) {
    if (!confirm('Delete this video from the index?')) return;
    try {
        await fetch(`${API}/api/videos/${id}`, { method: 'DELETE' });
        loadVideos();
    } catch (err) {
        console.error('Delete error:', err);
    }
}

// ---- Modal ----
function openModal(imgSrc, r) {
    modalImage.src = imgSrc;
    const matchInfo = r.match_type ? `<br><span style="color:var(--success)">Match: ${r.match_type}</span>` : '';
    modalInfo.innerHTML = `
        <strong>${r.video_filename}</strong> · ${formatTime(r.timestamp)}<br>
        ${r.caption ? `<br>${escapeHtml(r.caption)}` : ''}
        ${r.class_name ? `<br><span style="color:var(--accent)">${r.class_name}</span> (${(r.confidence * 100).toFixed(0)}%)` : ''}
        ${r.score ? `<br><span style="opacity:0.5">Score: ${(r.score * 100).toFixed(1)}%</span>` : ''}
        ${matchInfo}
    `;
    modalOverlay.classList.add('active');
}

function closeModal() {
    modalOverlay.classList.remove('active');
    modalImage.src = '';
}

// ---- Utils ----
function formatTime(seconds) {
    if (!seconds && seconds !== 0) return '0:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function escapeHtml(text) {
    if (!text) return '';
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}
