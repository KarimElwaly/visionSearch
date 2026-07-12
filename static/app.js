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
const cancelIndexBtn = document.getElementById('cancelIndexBtn');
const loadMoreBtn = document.getElementById('loadMoreBtn');
const filterVideoId = document.getElementById('filterVideoId');
const filterCamera = document.getElementById('filterCamera');
const filterClass = document.getElementById('filterClass');
const filterStart = document.getElementById('filterStart');
const filterEnd = document.getElementById('filterEnd');
const filterGroupTracks = document.getElementById('filterGroupTracks');

// ---- Pagination state ----
let currentTopK = 30;
const TOP_K_INCREMENT = 30;
let lastSearch = null;  // { kind: 'text' | 'image' | 'person', payload: ... }

// ---- State ----
let pollTimer = null;
let activeTab = 'text';
let pendingImageFile = null;
let pendingPersonFile = null;
let pendingVehicleFile = null;

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
        revokePreviewUrl(previewImg);
        imagePreview.style.display = 'none';
        imageDropZone.style.display = '';
        imageFileInput.value = '';
    });
    imageSearchBtn.addEventListener('click', doImageSearch);

    function setImageFile(file) {
        pendingImageFile = file;
        // Revoke any previous blob URL before assigning a new one so we don't
        // leak object URLs for each preview swap.
        revokePreviewUrl(previewImg);
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
        revokePreviewUrl(personPreviewImg);
        personPreview.style.display = 'none';
        personDropZone.style.display = '';
        personFileInput.value = '';
    });
    personSearchBtn.addEventListener('click', doPersonSearch);

    function setPersonFile(file) {
        pendingPersonFile = file;
        revokePreviewUrl(personPreviewImg);
        personPreviewImg.src = URL.createObjectURL(file);
        personPreview.style.display = 'flex';
        personDropZone.style.display = 'none';
    }

    // ---- Find Vehicle (Vehicle tab) ----
    const vehicleDropZone = document.getElementById('vehicleDropZone');
    const vehicleFileInput = document.getElementById('vehicleFileInput');
    const vehiclePreview = document.getElementById('vehiclePreview');
    const vehiclePreviewImg = document.getElementById('vehiclePreviewImg');
    const clearVehicle = document.getElementById('clearVehicle');
    const vehicleSearchBtn = document.getElementById('vehicleSearchBtn');

    vehicleDropZone.addEventListener('click', () => vehicleFileInput.click());
    vehicleDropZone.addEventListener('dragover', e => { e.preventDefault(); vehicleDropZone.classList.add('drag-over'); });
    vehicleDropZone.addEventListener('dragleave', () => vehicleDropZone.classList.remove('drag-over'));
    vehicleDropZone.addEventListener('drop', e => {
        e.preventDefault();
        vehicleDropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) setVehicleFile(file);
    });
    vehicleFileInput.addEventListener('change', e => {
        if (e.target.files[0]) setVehicleFile(e.target.files[0]);
    });
    clearVehicle.addEventListener('click', () => {
        pendingVehicleFile = null;
        revokePreviewUrl(vehiclePreviewImg);
        vehiclePreview.style.display = 'none';
        vehicleDropZone.style.display = '';
        vehicleFileInput.value = '';
    });
    vehicleSearchBtn.addEventListener('click', doVehicleSearch);

    function setVehicleFile(file) {
        pendingVehicleFile = file;
        revokePreviewUrl(vehiclePreviewImg);
        vehiclePreviewImg.src = URL.createObjectURL(file);
        vehiclePreview.style.display = 'flex';
        vehicleDropZone.style.display = 'none';
    }

    // ---- Index panel toggle ----
    toggleIndex.addEventListener('click', () => {
        toggleIndex.classList.toggle('active');
        indexPanel.classList.toggle('open');
    });
    indexBtn.addEventListener('click', startIndexing);
    if (cancelIndexBtn) cancelIndexBtn.addEventListener('click', cancelIndexing);
    if (loadMoreBtn) loadMoreBtn.addEventListener('click', loadMoreResults);

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

    currentTopK = 30;
    lastSearch = { kind: 'text', payload: query };
    searchBtn.innerHTML = '<span class="spinner"></span>';
    try {
        const mode = searchMode.value;
        const params = new URLSearchParams({ q: query, top_k: currentTopK, mode });
        appendFilterParams(params);
        const resp = await fetch(`${API}/api/search?${params.toString()}`);
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
    currentTopK = 30;
    lastSearch = { kind: 'image', payload: pendingImageFile };
    const btn = document.getElementById('imageSearchBtn');
    btn.innerHTML = '<span class="spinner"></span>';
    try {
        const formData = new FormData();
        formData.append('file', pendingImageFile);
        const params = new URLSearchParams({ top_k: currentTopK });
        appendFilterParams(params);
        const resp = await fetch(`${API}/api/search-by-image?${params.toString()}`, { method: 'POST', body: formData });
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
    currentTopK = 30;
    lastSearch = { kind: 'person', payload: pendingPersonFile };
    const btn = document.getElementById('personSearchBtn');
    btn.innerHTML = '<span class="spinner"></span>';
    try {
        const formData = new FormData();
        formData.append('file', pendingPersonFile);
        const params = new URLSearchParams({ top_k: currentTopK });
        appendFilterParams(params, { includeClass: false });
        const resp = await fetch(`${API}/api/search-by-person?${params.toString()}`, { method: 'POST', body: formData });
        const data = await resp.json();
        showResults(`Person search: "${pendingPersonFile.name}"`, data.count, data.results, true);
    } catch (err) {
        console.error('Person search failed:', err);
    } finally {
        btn.textContent = 'Find This Person';
    }
}

// ---- Vehicle Search ----
async function doVehicleSearch() {
    if (!pendingVehicleFile) return;
    currentTopK = 30;
    lastSearch = { kind: 'vehicle', payload: pendingVehicleFile };
    const btn = document.getElementById('vehicleSearchBtn');
    btn.innerHTML = '<span class="spinner"></span>';
    try {
        const formData = new FormData();
        formData.append('file', pendingVehicleFile);
        const params = new URLSearchParams({ top_k: currentTopK });
        appendFilterParams(params);
        const resp = await fetch(`${API}/api/search-by-vehicle?${params.toString()}`, { method: 'POST', body: formData });
        const data = await resp.json();
        showResults(`Vehicle search: "${pendingVehicleFile.name}"`, data.count, data.results, false);
    } catch (err) {
        console.error('Vehicle search failed:', err);
    } finally {
        btn.textContent = 'Find This Vehicle';
    }
}

// ---- Load more (pagination) ----
async function loadMoreResults() {
    if (!lastSearch) return;
    currentTopK += TOP_K_INCREMENT;
    loadMoreBtn.textContent = 'Loading...';
    try {
        let data;
        if (lastSearch.kind === 'text') {
            const mode = searchMode.value;
            const params = new URLSearchParams({ q: lastSearch.payload, top_k: currentTopK, mode });
            appendFilterParams(params);
            const resp = await fetch(`${API}/api/search?${params.toString()}`);
            data = await resp.json();
        } else if (lastSearch.kind === 'image') {
            const fd = new FormData(); fd.append('file', lastSearch.payload);
            const params = new URLSearchParams({ top_k: currentTopK });
            appendFilterParams(params);
            const resp = await fetch(`${API}/api/search-by-image?${params.toString()}`, { method: 'POST', body: fd });
            data = await resp.json();
        } else if (lastSearch.kind === 'person') {
            const fd = new FormData(); fd.append('file', lastSearch.payload);
            const params = new URLSearchParams({ top_k: currentTopK });
            appendFilterParams(params, { includeClass: false });
            const resp = await fetch(`${API}/api/search-by-person?${params.toString()}`, { method: 'POST', body: fd });
            data = await resp.json();
        } else {
            const fd = new FormData(); fd.append('file', lastSearch.payload);
            const params = new URLSearchParams({ top_k: currentTopK });
            appendFilterParams(params);
            const resp = await fetch(`${API}/api/search-by-vehicle?${params.toString()}`, { method: 'POST', body: fd });
            data = await resp.json();
        }
        resultsCount.textContent = `${data.count} found`;
        renderResults(data.results, lastSearch.kind === 'person');
    } catch (err) {
        console.error('Load more failed:', err);
    } finally {
        loadMoreBtn.textContent = 'Load more results';
    }
}

function appendFilterParams(params, opts = {}) {
    const includeClass = opts.includeClass !== false;
    if (filterVideoId && filterVideoId.value) params.set('video_id', filterVideoId.value);
    if (filterCamera && filterCamera.value.trim()) params.set('camera_id', filterCamera.value.trim());
    if (includeClass && filterClass && filterClass.value.trim()) params.set('class_name', filterClass.value.trim());
    if (filterStart && filterStart.value !== '') params.set('start_time', filterStart.value);
    if (filterEnd && filterEnd.value !== '') params.set('end_time', filterEnd.value);
    if (filterGroupTracks && filterGroupTracks.checked) params.set('group_tracks', 'true');
}

// ---- Cancel indexing ----
async function cancelIndexing() {
    cancelIndexBtn.disabled = true;
    cancelIndexBtn.textContent = 'Cancelling...';
    try {
        await fetch(`${API}/api/index/cancel`, { method: 'POST' });
    } catch (err) {
        console.error('Cancel failed:', err);
    }
}

// ---- Show Results ----
function showResults(title, count, results, isPersonSearch) {
    resultsSection.style.display = 'block';
    resultsTitle.textContent = title;
    resultsCount.textContent = `${count} found`;
    renderResults(results, isPersonSearch);
    // Show "Load more" if we may have more results server-side.
    if (loadMoreBtn) {
        loadMoreBtn.style.display = (results && results.length >= currentTopK) ? '' : 'none';
    }
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
        const trackBadge = r.track_id
            ? `<span class="source-badge">Track #${r.track_id}</span>`
            : '';
        const sourceBadge = isPersonSearch
            ? `${matchBadge}${trackBadge}`
            : `<span class="source-badge">${r.source}</span>${trackBadge}`;

        const clusterSize = (r._cluster && r._cluster.length) ? r._cluster.length : 0;
        const clusterBadge = clusterSize > 0
            ? `<span class="score-badge" style="top:2.2rem;background:#6366f1;">+${clusterSize}</span>`
            : '';

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
                    ${clusterBadge}
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
        if (cancelIndexBtn) { cancelIndexBtn.disabled = false; cancelIndexBtn.textContent = 'Cancel'; }
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
                        <div class="video-meta">${v.duration ? formatTime(v.duration) : 'N/A'} · ${v.total_frames || 0} frames · ${v.tracks || 0} tracks</div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <span class="video-status ${v.status}">${v.status}</span>
                    <button class="track-btn" onclick="viewVideoTracks(${v.id})" title="View tracks">Tracks</button>
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

async function viewVideoTracks(id) {
    try {
        const resp = await fetch(`${API}/api/videos/${id}/tracks?limit=200`);
        const data = await resp.json();
        if (!resp.ok) {
            alert(data.error || 'Could not load tracks');
            return;
        }
        lastSearch = null;
        const tracks = (data.tracks || []).map(t => ({
            type: 'track',
            source: 'track',
            track_id: t.id,
            frame_path: t.frame_path,
            crop_path: t.crop_path,
            timestamp: t.start_timestamp,
            frame_number: t.start_frame_number,
            video_filename: t.video_filename,
            video_path: t.video_path,
            class_name: t.class_name,
            confidence: t.best_confidence || 0,
            score: t.best_confidence || 0,
            detection_count: t.detection_count || 0,
            track_duration: t.duration || 0,
            caption: `${t.class_name || 'object'} track · ${t.detection_count || 0} detections · ${formatTime(t.start_timestamp)}-${formatTime(t.end_timestamp)}`,
            bbox: t.bbox_x1 !== null ? [t.bbox_x1, t.bbox_y1, t.bbox_x2, t.bbox_y2] : null,
        }));
        showResults(`Tracks in ${data.video.filename}`, data.count, tracks, false);
        if (loadMoreBtn) loadMoreBtn.style.display = 'none';
    } catch (err) {
        console.error('Load tracks error:', err);
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
    const trackInfo = r.track_id ? `<br><span style="color:var(--warning)">Track #${r.track_id}${r.detection_count ? ` · ${r.detection_count} detections` : ''}</span>` : '';
    modalInfo.innerHTML = `
        <strong>${r.video_filename}</strong> · ${formatTime(r.timestamp)}<br>
        ${r.caption ? `<br>${escapeHtml(r.caption)}` : ''}
        ${r.class_name ? `<br><span style="color:var(--accent)">${r.class_name}</span> (${(r.confidence * 100).toFixed(0)}%)` : ''}
        ${r.score ? `<br><span style="opacity:0.5">Score: ${(r.score * 100).toFixed(1)}%</span>` : ''}
        ${matchInfo}
        ${trackInfo}
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
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) {
        return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function escapeHtml(text) {
    if (!text) return '';
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

/**
 * Release the blob URL currently assigned to an <img> preview so the browser
 * can free its underlying Blob. Called before assigning a new preview image
 * and when clearing the preview, preventing an object-URL leak per upload.
 */
function revokePreviewUrl(imgEl) {
    if (!imgEl) return;
    const src = imgEl.src;
    if (src && src.startsWith('blob:')) {
        try { URL.revokeObjectURL(src); } catch (_) { /* ignore */ }
    }
    imgEl.removeAttribute('src');
}
