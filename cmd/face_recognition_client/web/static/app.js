'use strict';

// Edge AI – Face Recognition — page logic.
//
// The page shows the camera as it arrives and draws the boxes itself, from the
// results the app publishes for the frames it has analysed.

const $ = (id) => document.getElementById(id);

const tabLive = $('tabLive'), tabEnroll = $('tabEnroll'), peopleCount = $('peopleCount');
const stateEl = $('state'), stateText = $('stateText');
const screenEl = $('screen'), video = $('video'), overlay = $('overlay');
const placeholder = $('placeholder'), recording = $('recording'), recordingText = $('recordingText');
const cameraSelect = $('camera'), btnPause = $('btnPause'), stats = $('stats');
const panelLive = $('panelLive'), panelEnroll = $('panelEnroll');
const identify = $('identify'), facesEl = $('faces'), facesEmpty = $('facesEmpty'), liveHint = $('liveHint');
const enrollForm = $('enrollForm'), enrollName = $('enrollName'), knownNames = $('knownNames');
const enrollBar = $('enrollBar'), enrollStatus = $('enrollStatus');
const btnRecord = $('btnRecord'), btnCancel = $('btnCancel');
const peopleEl = $('people'), peopleEmpty = $('peopleEmpty');
const threshold = $('threshold'), thresholdValue = $('thresholdValue'), thresholdHint = $('thresholdHint');
const tipRecordings = $('tipRecordings');
const toast = $('toast');

const COLOURS = { known: '#34d399', unknown: '#fbbf24', face: '#8f7bea', target: '#8f7bea', other: '#6b6b88' };
const RESULTS_EVERY_MS = 120;
const STATUS_EVERY_MS = 1000;
const STATUS_ENROLLING_MS = 250;

let section = 'live';
let paused = false;
let status = null;
let enrolling = false;            // a recording started from this page is running
let enrollSeq = null;             // number of the last recording outcome already shown
let frame = { width: 640, height: 480 };
let mode = 'recognize';
let faces = new Map();            // face number -> { box drawn now, box to reach, result }
let facesSignature = '';

// ---------------------------------------------------------------- app requests

async function api(path, body) {
  const options = body === undefined
    ? { cache: 'no-store' }
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) throw new Error(data.error || ('HTTP ' + response.status));
  return data;
}

let toastTimer = 0;
function showToast(message, isError) {
  toast.textContent = message;
  toast.className = 'toast' + (isError ? ' error' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add('hidden'), 4000);
}

// ---------------------------------------------------------------- camera stream

let retryTimer = 0, retryDelay = 1000;

function startStream() {
  clearTimeout(retryTimer);
  if (paused || document.hidden) return;
  video.src = 'video_feed?t=' + Date.now();
}

function stopStream() {
  clearTimeout(retryTimer);
  video.removeAttribute('src');
  video.src = '';                               // closes the request, so the app can release the camera
  faces.clear();
  showPlaceholder(paused ? 'Paused.' : 'Connecting to the camera…');
}

function showPlaceholder(message) {
  placeholder.textContent = message;
  placeholder.classList.remove('hidden');
}

video.addEventListener('load', () => {
  retryDelay = 1000;
  placeholder.classList.add('hidden');
});

video.addEventListener('error', () => {
  if (paused || document.hidden || !video.getAttribute('src')) return;
  showPlaceholder((status && status.camera_error) ? 'Camera: ' + status.camera_error : 'Connecting to the camera…');
  retryTimer = setTimeout(startStream, retryDelay);
  retryDelay = Math.min(retryDelay * 2, 4000);
});

btnPause.addEventListener('click', () => {
  paused = !paused;
  btnPause.textContent = paused ? 'Resume' : 'Pause';
  if (paused) {
    if (enrolling) cancelEnroll();
    stopStream();
  } else {
    showPlaceholder('Connecting to the camera…');
    startStream();
  }
  showState();
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopStream(); else startStream();
});

// ---------------------------------------------------------------- state

function showState() {
  let kind = 'busy', text = 'Starting…';
  if (paused) {
    kind = ''; text = 'Paused';
  } else if (!status) {
    text = 'Connecting to the app…';
  } else if (status.models_error) {
    kind = 'error'; text = 'Models: ' + status.models_error;
  } else if (status.camera_connected) {
    kind = 'ok'; text = 'Camera ' + (status.camera_index || 'connected');
  } else if (status.camera_error) {
    const looking = status.camera_error.startsWith('looking');
    kind = looking ? 'busy' : 'error';
    text = looking ? 'Looking for cameras…' : 'Camera: ' + status.camera_error;
  } else if (!status.models_ready) {
    text = 'Loading the models…';
  }
  stateEl.className = 'state ' + kind;
  stateText.textContent = text;
  stateEl.title = text;
  if (!paused && !placeholder.classList.contains('hidden') && kind !== 'ok') showPlaceholder(text);
}

function showCameras() {
  const devices = status.camera_devices || [];
  const signature = devices.map((d) => d.device_id + '|' + d.label).join('\n');
  if (cameraSelect.dataset.signature !== signature) {
    cameraSelect.dataset.signature = signature;
    cameraSelect.textContent = '';
    for (const d of devices) cameraSelect.add(new Option(d.label || d.device_id, d.device_id));
    if (!devices.length) cameraSelect.add(new Option('No camera found', ''));
  }
  if (document.activeElement !== cameraSelect && status.camera_index) cameraSelect.value = status.camera_index;
  cameraSelect.disabled = devices.length < 2 || enrolling;
}

cameraSelect.addEventListener('change', async () => {
  if (!cameraSelect.value) return;
  try {
    await api('api/camera/select', { device_id: cameraSelect.value });
    showPlaceholder('Switching camera…');
  } catch (e) {
    showToast('Could not switch camera: ' + e.message, true);
  }
});

function showStats() {
  if (paused || !status.camera_connected) { stats.textContent = ''; return; }
  const fps = Number(status.fps) || 0, ms = Math.round(Number(status.infer_ms) || 0);
  stats.textContent = fps.toFixed(1) + ' fps · ' + ms + ' ms';
}

async function refreshStatus() {
  try {
    status = await api('api/status');
  } catch (e) {
    status = null;
    stateEl.className = 'state error';
    stateText.textContent = 'The app is not answering';
    return;
  }
  mode = status.mode || mode;
  peopleCount.textContent = String(status.people || 0);
  if (mode !== 'enroll' && document.activeElement !== identify) identify.checked = mode === 'recognize';
  showState();
  showCameras();
  showStats();
  showEnrollProgress();
  liveHint.textContent = (identify.checked && !status.people) ? 'Nobody is enrolled yet. Open Enroll to add the first person.' : '';
}

async function statusLoop() {
  await refreshStatus();
  setTimeout(statusLoop, enrolling ? STATUS_ENROLLING_MS : STATUS_EVERY_MS);
}

// ---------------------------------------------------------------- results and drawing

async function resultsLoop() {
  if (!paused && !document.hidden) {
    try {
      const data = await api('api/results');
      if (data.frame_width > 0 && data.frame_height > 0) {
        if (data.frame_width !== frame.width || data.frame_height !== frame.height) {
          frame = { width: data.frame_width, height: data.frame_height };
          screenEl.style.setProperty('--ratio', String(frame.width / frame.height));
        }
      }
      mode = data.mode || mode;
      takeResults(data.results || []);
    } catch (e) { /* the status shows what is wrong */ }
  }
  setTimeout(resultsLoop, RESULTS_EVERY_MS);
}

function takeResults(results) {
  const seen = new Set();
  for (const r of results) {
    const d = r.detection;
    const box = { x: d.xmin, y: d.ymin, w: d.xmax - d.xmin, h: d.ymax - d.ymin };
    const known = faces.get(r.id);
    if (known) {
      known.to = box;
      known.result = r;
    } else {
      faces.set(r.id, { now: { ...box }, to: box, result: r });
    }
    seen.add(r.id);
  }
  for (const id of [...faces.keys()]) if (!seen.has(id)) faces.delete(id);
  if (section === 'live') showFaces(results);
}

// How a face is shown: its colour, its label on the picture, its line in the list.
function describe(r) {
  const detection = Math.round((r.detection.score || 0) * 100);
  if (mode === 'enroll') {
    return r.target ? { kind: 'target', label: 'Recording' } : { kind: 'other', label: '' };
  }
  if (mode !== 'recognize') {
    return { kind: 'face', label: 'Face ' + detection + '%', name: 'Face', value: r.detection.score, detail: 'Identification is off' };
  }
  const m = r.match;
  if (m && m.accepted) {
    const pct = Math.round(m.similarity * 100);
    return { kind: 'known', label: m.name + ' ' + pct + '%', name: m.name, value: m.similarity, detail: 'Match ' + pct + '%' };
  }
  const closest = (m && m.candidate) ? 'Closest: ' + m.candidate + ' ' + Math.round(m.similarity * 100) + '%' : 'Not enrolled';
  return { kind: 'unknown', label: 'Unknown', name: 'Unknown', value: m ? m.similarity : 0, detail: closest };
}

function initials(name) {
  return name.split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0].toUpperCase()).join('') || '?';
}

function showFaces(results) {
  const rows = results.map((r) => ({ id: r.id, ...describe(r) })).filter((row) => row.name);
  const signature = rows.map((row) => [row.id, row.kind, row.name, row.detail].join('|')).join('\n');
  facesEmpty.classList.toggle('hidden', rows.length > 0);
  if (signature !== facesSignature) {
    facesSignature = signature;
    facesEl.textContent = '';
    for (const row of rows) {
      const li = document.createElement('li');
      li.className = 'face' + (row.kind === 'known' ? ' known' : '');
      li.dataset.id = row.id;
      const avatar = document.createElement('span');
      avatar.className = 'avatar';
      avatar.textContent = row.kind === 'known' ? initials(row.name) : '?';
      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = row.name;
      const detail = document.createElement('span');
      detail.className = 'detail';
      detail.textContent = row.detail;
      const meter = document.createElement('span');
      meter.className = 'meter';
      meter.appendChild(document.createElement('i'));
      li.append(avatar, name, detail, meter);
      facesEl.appendChild(li);
    }
  }
  for (const row of rows) {
    const bar = facesEl.querySelector('.face[data-id="' + row.id + '"] .meter i');
    if (bar) bar.style.width = Math.max(0, Math.min(100, Math.round((row.value || 0) * 100))) + '%';
  }
}

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const width = screenEl.clientWidth, height = screenEl.clientHeight;
  if (overlay.width !== Math.round(width * dpr) || overlay.height !== Math.round(height * dpr)) {
    overlay.width = Math.round(width * dpr);
    overlay.height = Math.round(height * dpr);
  }
  const ctx = overlay.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  if (!paused && placeholder.classList.contains('hidden')) {
    // The picture keeps its proportions inside the box; the boxes follow it.
    const scale = Math.min(width / frame.width, height / frame.height);
    const left = (width - frame.width * scale) / 2, top = (height - frame.height * scale) / 2;
    ctx.font = '600 13px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    for (const face of faces.values()) {
      for (const k of ['x', 'y', 'w', 'h']) face.now[k] += (face.to[k] - face.now[k]) * 0.35;   // glide to the new place
      const look = describe(face.result);
      const colour = COLOURS[look.kind];
      const x = left + face.now.x * scale, y = top + face.now.y * scale;
      const w = face.now.w * scale, h = face.now.h * scale;
      ctx.lineWidth = look.kind === 'other' ? 1.5 : look.kind === 'target' ? 3.5 : 2.5;
      ctx.strokeStyle = colour;
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, 8);
      ctx.stroke();
      if (look.label) {
        const textWidth = ctx.measureText(look.label).width;
        const labelY = y > 30 ? y - 26 : y + 6;
        ctx.fillStyle = colour;
        ctx.beginPath();
        ctx.roundRect(x, labelY, textWidth + 16, 22, 6);
        ctx.fill();
        ctx.fillStyle = '#07070c';
        ctx.fillText(look.label, x + 8, labelY + 11.5);
      }
    }
  }
  requestAnimationFrame(draw);
}

identify.addEventListener('change', async () => {
  try {
    await api('api/mode', { mode: identify.checked ? 'recognize' : 'detect' });
  } catch (e) {
    showToast(e.message, true);
  }
  refreshStatus();
});

// ---------------------------------------------------------------- match threshold

// The app answers with the rules in force: the threshold, and the higher one
// used while only one person is enrolled.
function showThreshold(rules) {
  const pct = Math.round(rules.match_threshold * 100);
  threshold.min = Math.round(rules.match_threshold_min * 100);
  threshold.max = Math.round(rules.match_threshold_max * 100);
  threshold.value = pct;
  threshold.disabled = false;
  thresholdValue.textContent = pct + '%';
  thresholdHint.textContent = 'A face gets a name from ' + pct + '% similarity (' + Math.round(rules.single_threshold * 100) +
    '% while only one person is enrolled). The same person usually scores 50 to 80%, another person below 35%. ' +
    'Lower it if enrolled people often show as Unknown; raise it if someone gets the wrong name.';
}

async function loadThreshold() {
  try {
    showThreshold(await api('api/settings'));
  } catch (e) {
    thresholdHint.textContent = 'Could not read the match threshold: ' + e.message;
  }
}

threshold.addEventListener('input', () => { thresholdValue.textContent = threshold.value + '%'; });

threshold.addEventListener('change', async () => {
  try {
    showThreshold(await api('api/settings', { match_threshold: Number(threshold.value) / 100 }));
  } catch (e) {
    showToast('Could not change the match threshold: ' + e.message, true);
    loadThreshold();
  }
});

// ---------------------------------------------------------------- sections

function showSection(name) {
  if (section === name) return;
  if (enrolling) cancelEnroll();
  section = name;
  tabLive.classList.toggle('active', name === 'live');
  tabEnroll.classList.toggle('active', name === 'enroll');
  tabLive.setAttribute('aria-pressed', String(name === 'live'));
  tabEnroll.setAttribute('aria-pressed', String(name === 'enroll'));
  panelLive.classList.toggle('hidden', name !== 'live');
  panelEnroll.classList.toggle('hidden', name !== 'enroll');
  if (name === 'enroll') {
    refreshPeople();
    enrollName.focus();
  }
}

tabLive.addEventListener('click', () => showSection('live'));
tabEnroll.addEventListener('click', () => showSection('enroll'));

// ---------------------------------------------------------------- enrolling

function setEnrollStatus(text, kind) {
  enrollStatus.textContent = text;
  enrollStatus.className = 'enroll-status' + (kind ? ' ' + kind : '');
}

function setEnrolling(on) {
  enrolling = on;
  btnRecord.disabled = on;
  enrollName.disabled = on;
  btnCancel.classList.toggle('hidden', !on);
  recording.classList.toggle('hidden', !on);
  if (!on) enrollBar.style.width = '0';
}

enrollForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const name = enrollName.value.trim();
  if (!name) { setEnrollStatus('Type the name of the person first.', 'error'); enrollName.focus(); return; }
  if (paused) { setEnrollStatus('The camera is paused. Press Resume first.', 'error'); return; }
  if (!status || !status.camera_connected) { setEnrollStatus('The camera is not ready yet.', 'error'); return; }
  try {
    enrollSeq = status.enroll_seq || 0;
    await api('api/enroll/start', { name: name });
    setEnrolling(true);
    recordingText.textContent = 'Recording ' + name;
    setEnrollStatus('Recording… look at the camera.');
    refreshStatus();
  } catch (e) {
    setEnrollStatus('Could not start: ' + e.message, 'error');
  }
});

async function cancelEnroll() {
  setEnrolling(false);
  setEnrollStatus('Recording cancelled.');
  try { await api('api/enroll/cancel', {}); } catch (e) { /* nothing to undo */ }
}

btnCancel.addEventListener('click', cancelEnroll);

function showEnrollProgress() {
  if (!enrolling || !status) return;
  // The app closes the recording by itself and says how it went.
  if ((status.enroll_seq || 0) !== enrollSeq) {
    enrollSeq = status.enroll_seq;
    setEnrolling(false);
    if (status.enroll_result === 'saved') {
      setEnrollStatus('Saved ' + status.enroll_result_name + ' — ' + status.enroll_result_samples + ' samples.', 'ok');
      showToast(status.enroll_result_name + ' is enrolled.');
      enrollName.value = '';
      refreshPeople();
    } else if (status.enroll_result === 'no_face') {
      setEnrollStatus('No face was seen. Try again, closer to the camera and with more light.', 'error');
    } else {
      setEnrollStatus('The recording could not be saved. Try again.', 'error');
    }
    return;
  }
  const duration = Number(status.enroll_duration_sec) || 5, elapsed = Number(status.enroll_elapsed_sec) || 0;
  enrollBar.style.width = Math.min(100, Math.round(elapsed / duration * 100)) + '%';
  const left = Math.max(0, Math.ceil(duration - elapsed));
  const samples = status.enroll_samples || 0;
  setEnrollStatus(samples
    ? 'Recording… ' + left + ' s left, ' + samples + ' samples.'
    : 'Recording… ' + left + ' s left. No face seen yet: look at the camera.');
}

// ---------------------------------------------------------------- enrolled people

function when(iso) {
  const date = new Date(iso);
  if (!iso || isNaN(date)) return '';
  return date.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) + ', ' +
         date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

async function refreshPeople() {
  let people = [], most = 0;
  try {
    const data = await api('api/people');
    people = data.people || [];
    most = data.max_recordings || 0;
  } catch (e) {
    showToast('Could not read the enrolled people: ' + e.message, true);
    return;
  }
  people.sort((a, b) => a.name.localeCompare(b.name));
  peopleCount.textContent = String(people.length);
  peopleEmpty.classList.toggle('hidden', people.length > 0);
  if (most) tipRecordings.textContent = 'Record the same name again in other light (daylight, lamp light): the last ' + most + ' recordings are kept, and a face is compared with each.';
  peopleEl.textContent = '';
  knownNames.textContent = '';
  for (const person of people) {
    const li = document.createElement('li');
    li.className = 'person';
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = person.name;
    const detail = document.createElement('span');
    detail.className = 'detail';
    const recordings = person.recordings || 0;
    if (!recordings) {
      li.classList.add('outdated');
      detail.textContent = 'Recorded with an earlier version — record again';
    } else {
      detail.textContent = recordings + (most ? ' of ' + most : '') + (recordings === 1 && !most ? ' recording' : ' recordings') +
        (person.updated_at ? ' · ' + when(person.updated_at) : '');
    }
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'quiet';
    remove.textContent = 'Remove';
    remove.addEventListener('click', () => removePerson(person.name));
    li.append(name, remove, detail);
    peopleEl.appendChild(li);
    knownNames.appendChild(new Option(person.name));
  }
}

async function removePerson(name) {
  if (!confirm('Remove ' + name + ' from the enrolled people?')) return;
  try {
    await api('api/remove', { name: name });
    showToast(name + ' was removed.');
  } catch (e) {
    showToast('Could not remove ' + name + ': ' + e.message, true);
  }
  refreshPeople();
}

// ---------------------------------------------------------------- start

screenEl.style.setProperty('--ratio', String(frame.width / frame.height));
startStream();
loadThreshold();
statusLoop();
resultsLoop();
requestAnimationFrame(draw);
