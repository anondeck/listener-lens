function setupPlayer(player) {
  const audio = player.querySelector('audio');
  const choices = [...player.querySelectorAll('.ab')];
  let selected = choices.find(choice => choice.classList.contains('active')) || choices[0];

  function choose(choice) {
    selected = choice;
    choices.forEach(item => {
      const active = item === choice;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    audio.pause();
    if (choice.dataset.audio) audio.src = choice.dataset.audio;
    if (choice.dataset.side && player.audioUrls) audio.src = player.audioUrls[choice.dataset.side];
    audio.load();
    choices.forEach(item => item.classList.remove('playing'));
  }

  choices.forEach(choice => choice.addEventListener('click', async () => {
    const alreadySelected = selected === choice;
    if (!alreadySelected) choose(choice);
    if (alreadySelected && !audio.paused) {
      audio.pause();
      choice.classList.remove('playing');
      return;
    }
    if (audio.src) {
      try {
        await audio.play();
        choice.classList.add('playing');
      } catch {
        choice.classList.remove('playing');
      }
    }
  }));
  audio.addEventListener('timeupdate', () => {
    const side = selected?.dataset.side;
    if (!side || !Number.isFinite(audio.duration) || audio.duration <= 0) return;
    const progress = player.querySelector(`[data-track-progress="${side}"]`);
    const time = player.querySelector(`[data-track-time="${side}"]`);
    if (progress) progress.value = audio.currentTime / audio.duration;
    if (time) time.textContent = `${formatClock(audio.currentTime)} / ${formatClock(audio.duration)}`;
  });
  audio.addEventListener('ended', () => {
    selected?.classList.remove('playing');
    const side = selected?.dataset.side;
    const progress = side ? player.querySelector(`[data-track-progress="${side}"]`) : null;
    if (progress) progress.value = 0;
  });
  return { choose };
}

const playerControllers = new Map();
document.querySelectorAll('[data-player]').forEach(player => playerControllers.set(player, setupPlayer(player)));

function audioUrl(base64) {
  return URL.createObjectURL(new Blob([wavBytes(base64)], { type: 'audio/wav' }));
}

function wavBytes(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function formatClock(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safe / 60);
  const remainder = Math.floor(safe % 60);
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
}

function drawWaveform(canvas, samples, color) {
  const context = canvas.getContext('2d');
  if (!context) return;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  context.clearRect(0, 0, width, height);
  context.strokeStyle = color;
  context.lineWidth = Math.max(1, ratio);
  context.beginPath();
  const center = height / 2;
  const stride = Math.max(1, Math.floor(samples.length / width));
  for (let x = 0; x < width; x += 1) {
    const start = x * stride;
    let low = 1;
    let high = -1;
    for (let index = start; index < Math.min(start + stride, samples.length); index += 1) {
      low = Math.min(low, samples[index]);
      high = Math.max(high, samples[index]);
    }
    const y1 = center + low * center * .86;
    const y2 = center + high * center * .86;
    context.moveTo(x + .5, y1);
    context.lineTo(x + .5, y2);
  }
  context.stroke();
}

function decodedWavChannel(buffer) {
  const view = new DataView(buffer);
  if (view.byteLength < 44 || view.getUint32(0, false) !== 0x52494646 || view.getUint32(8, false) !== 0x57415645) {
    throw new Error('Unsupported WAV container');
  }
  let offset = 12;
  let format = null;
  let dataOffset = null;
  let dataSize = null;
  while (offset + 8 <= view.byteLength) {
    const chunkId = view.getUint32(offset, false);
    const chunkSize = view.getUint32(offset + 4, true);
    const start = offset + 8;
    if (chunkId === 0x666d7420 && chunkSize >= 16) {
      format = {
        audioFormat: view.getUint16(start, true),
        channels: view.getUint16(start + 2, true),
        sampleRate: view.getUint32(start + 4, true),
        blockAlign: view.getUint16(start + 12, true),
        bits: view.getUint16(start + 14, true)
      };
    } else if (chunkId === 0x64617461) {
      dataOffset = start;
      dataSize = Math.min(chunkSize, view.byteLength - start);
      break;
    }
    offset = start + chunkSize + (chunkSize % 2);
  }
  if (!format || dataOffset === null || dataSize === null || !format.blockAlign) {
    throw new Error('Incomplete WAV');
  }
  const frameCount = Math.floor(dataSize / format.blockAlign);
  const samples = new Float32Array(frameCount);
  const bytesPerSample = format.bits / 8;
  for (let index = 0; index < frameCount; index += 1) {
    const position = dataOffset + index * format.blockAlign;
    if (format.audioFormat === 3 && format.bits === 32) samples[index] = view.getFloat32(position, true);
    else if (format.bits === 8) samples[index] = (view.getUint8(position) - 128) / 128;
    else if (format.bits === 16) samples[index] = view.getInt16(position, true) / 32768;
    else if (format.bits === 24) {
      let value = view.getUint8(position) | (view.getUint8(position + 1) << 8) | (view.getUint8(position + 2) << 16);
      if (value & 0x800000) value |= 0xff000000;
      samples[index] = value / 8388608;
    } else if (format.bits === 32) samples[index] = view.getInt32(position, true) / 2147483648;
    else throw new Error(`Unsupported ${format.bits}-bit WAV`);
    if (bytesPerSample > format.blockAlign) throw new Error('Invalid WAV alignment');
  }
  return { samples, duration: frameCount / format.sampleRate };
}

function paintPlayerWaveforms(player, wavBySide) {
  const colors = { neutral: '#2743b5', lens: '#dd4426', gibberish: '#dd4426', speaker: '#1d1c1a' };
  try {
    [...player.querySelectorAll('[data-waveform-side]')].forEach(canvas => {
      const side = canvas.dataset.waveformSide;
      const base64 = wavBySide?.[side];
      if (!base64) return;
      const bytes = wavBytes(base64);
      const decoded = decodedWavChannel(bytes.buffer);
      drawWaveform(canvas, decoded.samples, colors[side]);
      const time = player.querySelector(`[data-track-time="${side}"]`);
      if (time) time.textContent = `00:00 / ${formatClock(decoded.duration)}`;
    });
  } catch {
    // Audio remains playable if a future renderer returns an unsupported WAV encoding.
  }
}

// ---- Listener-lens: Azure SSML lane ----
const lensForm = document.querySelector('#lens-form');
const lensStatus = document.querySelector('#lens-status');
const lensResult = document.querySelector('#lens-result');
const runtimePlayer = document.querySelector('#runtime-player');
const sourceLanguage = document.querySelector('#source-language');
const listenerProfile = document.querySelector('#listener-profile');
const sourceText = document.querySelector('#source-text');
const sourceCharacterCount = document.querySelector('#source-character-count');
const lensSubmit = lensForm.querySelector('.lens-submit');
const evidenceDrawer = document.querySelector('#evidence-drawer');
const activityForm = document.querySelector('#activity-form');
const activityOutput = document.querySelector('#activity-output');
const activitySubmit = activityForm.querySelector('button[type="submit"]');
let currentActivityMetadata = null;
let runtimeUrls = [];

document.querySelectorAll('[data-open-evidence]').forEach(button => {
  button.addEventListener('click', () => {
    if (typeof evidenceDrawer.showModal === 'function') evidenceDrawer.showModal();
    else evidenceDrawer.setAttribute('open', '');
  });
});
document.querySelector('[data-close-evidence]').addEventListener('click', () => evidenceDrawer.close());
evidenceDrawer.addEventListener('click', event => {
  if (event.target === evidenceDrawer) evidenceDrawer.close();
});

// Two modes, two claims. Listener Lens is cross-language recategorization.
// Sound Minus Meaning is source-only gibberish and never accepts a listener.
let mode = 'lens';
const modeButtons = [...lensForm.querySelectorAll('.mode')];
const firstSide = runtimePlayer.querySelector('[data-side="neutral"]');
const secondSide = runtimePlayer.querySelector('[data-side="lens"], [data-side="gibberish"]');

function labelSide(button, label) {
  button.dataset.label = `${button.dataset.side === 'neutral' ? 'A' : 'B'} · ${label}`;
  const visible = button.querySelector('[data-side-label]');
  if (visible) visible.textContent = `Play ${button.dataset.side === 'neutral' ? 'A' : 'B'}`;
}

function applyGibberishCopy() {
  const state = mode === 'gibberish' ? 'source' : 'none';
  document.querySelectorAll('[data-gib-copy]').forEach(node => {
    node.hidden = node.dataset.gibCopy !== state;
  });
  if (mode === 'gibberish') {
    document.querySelector('#route-listener-name').textContent = 'SOURCE SOUND ONLY';
  }
  if (mode !== 'gibberish') return;
  labelSide(firstSide, 'as you say it');
  labelSide(secondSide, 'your language, no meaning');
}

function submitLabel() {
  return mode === 'gibberish'
    ? 'Generate sound-only version'
    : 'Generate listener comparison';
}

function updateSourceCharacterCount() {
  sourceCharacterCount.textContent = String(sourceText.value.length);
}

sourceText.addEventListener('input', updateSourceCharacterCount);
updateSourceCharacterCount();

function applyMode(next) {
  mode = next;
  lensForm.dataset.mode = next;
  modeButtons.forEach(button => {
    const active = button.dataset.mode === next;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  document.querySelectorAll('[data-lens-only]').forEach(node => { node.hidden = next !== 'lens'; });
  document.querySelectorAll('[data-gibberish-only]').forEach(node => { node.hidden = next === 'lens'; });
  const gibberish = next === 'gibberish';
  secondSide.dataset.side = gibberish ? 'gibberish' : 'lens';
  secondSide.closest('.audio-track').dataset.track = gibberish ? 'gibberish' : 'lens';
  runtimePlayer.querySelectorAll('[data-waveform-side="lens"], [data-track-progress="lens"], [data-track-time="lens"]').forEach(node => {
    node.hidden = gibberish;
  });
  runtimePlayer.querySelectorAll('[data-waveform-side="gibberish"], [data-track-progress="gibberish"], [data-track-time="gibberish"]').forEach(node => {
    node.hidden = !gibberish;
  });
  labelSide(firstSide, 'as you say it');
  labelSide(secondSide, gibberish ? 'your language, no meaning' : 'as they hear it');
  lensSubmit.textContent = submitLabel();
  if (!gibberish) {
    const listener = listenersFor(sourceLanguage.value)
      .find(row => row.listener === lastListenerLocale);
    if (listener) selectListener(listener, false);
  }
  applyGibberishCopy();
  // A result built under the other mode describes something the page is no
  // longer offering, so it goes rather than sitting there mislabelled.
  lensResult.hidden = true;
  runtimePlayer.hidden = true;
  releaseRuntimeAudio();
  lensStatus.className = 'lens-status';
  resetDrillSetup();
  setLensStatus('Ready.', gibberish
    ? 'Type in the source language and generate a meaning-free version with the same word and syllable structure.'
    : 'Try the example, or enter one or two short sentences.');
}

modeButtons.forEach(button =>
  button.addEventListener('click', () => applyMode(button.dataset.mode)));

// The full direction set is generated from the same registry the Worker
// validates against, so the menu cannot drift from what the service accepts.
import {
  LANGUAGE_NAMES, NATIVE_NAMES, LANGUAGE_FAMILIES,
  EXAMPLE_SENTENCE, LISTENERS_BY_SOURCE, DIRECTION_SUGGESTIONS
} from '/listener-directions.generated.js';

const allExamples = new Set(Object.values(EXAMPLE_SENTENCE));

function setLensStatus(heading, message) {
  const strong = document.createElement('strong');
  strong.textContent = heading;
  lensStatus.replaceChildren(strong, document.createTextNode(` ${message}`));
}

function listenersFor(locale) {
  return LISTENERS_BY_SOURCE[locale] || LISTENERS_BY_SOURCE['en-US'];
}

// A 30-item select shows about eight rows at a time. Rendering every language
// as a chip, grouped by family, puts the whole set in view: if a listener
// disappoints, its relatives behave alike and are already adjacent.
function renderPicker(picker, { items, selected, onPick, lead }) {
  const groups = picker.querySelector('.picker-groups');
  const filter = picker.querySelector('.picker-search').value.trim().toLowerCase();
  const byLocale = new Map(items.map(item => [item.locale, item]));
  groups.replaceChildren();

  if (lead) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'picker-chip';
    chip.setAttribute('role', 'option');
    chip.setAttribute('aria-selected', String(selected === null));
    chip.textContent = lead.label;
    chip.title = lead.title;
    chip.addEventListener('click', () => onPick({ locale: null }));
    const row = document.createElement('div');
    row.className = 'picker-chips';
    row.append(chip);
    groups.append(row);
  }

  for (const { family, locales } of LANGUAGE_FAMILIES) {
    const matches = locales
      .map(locale => byLocale.get(locale))
      .filter(item => item && (!filter
        || (LANGUAGE_NAMES[item.locale] || '').toLowerCase().includes(filter)
        || (NATIVE_NAMES[item.locale] || '').toLowerCase().includes(filter)));
    if (!matches.length) continue;

    const heading = document.createElement('p');
    heading.className = 'picker-family';
    heading.textContent = family;
    const row = document.createElement('div');
    row.className = 'picker-chips';

    for (const item of matches) {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'picker-chip';
      chip.setAttribute('role', 'option');
      chip.setAttribute('aria-selected', String(item.locale === selected));
      if (item.curated) chip.dataset.curated = 'true';
      // Endonym first: someone looking for Czech scans for "Čeština".
      chip.textContent = NATIVE_NAMES[item.locale] || LANGUAGE_NAMES[item.locale];
      chip.title = LANGUAGE_NAMES[item.locale] || item.locale;
      chip.addEventListener('click', () => onPick(item));
      row.append(chip);
    }
    groups.append(heading, row);
  }

  // The lead chip is not a language, so it must not answer for one: a search
  // that matches nothing still says so.
  if (groups.childElementCount === (lead ? 1 : 0)) {
    const empty = document.createElement('p');
    empty.className = 'picker-note';
    empty.textContent = 'No language matches that search.';
    groups.append(empty);
  }
}

function setupPicker(name) {
  const picker = document.querySelector(`[data-picker="${name}"]`);
  const toggle = picker.querySelector('.picker-toggle');
  const panel = picker.querySelector('.picker-panel');
  const search = picker.querySelector('.picker-search');
  const current = picker.querySelector('[data-picker-current]');

  const close = () => { panel.hidden = true; toggle.setAttribute('aria-expanded', 'false'); };
  toggle.addEventListener('click', () => {
    const opening = panel.hidden;
    document.querySelectorAll('.picker-panel').forEach(other => {
      other.hidden = true;
      other.closest('.picker').querySelector('.picker-toggle')
        .setAttribute('aria-expanded', 'false');
    });
    if (!opening) return;
    panel.hidden = false;
    toggle.setAttribute('aria-expanded', 'true');
    search.value = '';
    picker.render();
    search.focus();
  });
  search.addEventListener('input', () => picker.render());
  document.addEventListener('click', event => {
    if (!picker.contains(event.target)) close();
  });
  picker.addEventListener('keydown', event => {
    if (event.key === 'Escape') { close(); toggle.focus(); }
  });
  return { picker, current, close };
}

const sourcePicker = setupPicker('source');
const listenerPicker = setupPicker('listener');

function selectListener(row, replaceExample = true) {
  listenerProfile.value = row.profileId;
  listenerPicker.current.textContent =
    NATIVE_NAMES[row.listener] || LANGUAGE_NAMES[row.listener];
  document.querySelector('#route-listener-name').textContent =
    (NATIVE_NAMES[row.listener] || LANGUAGE_NAMES[row.listener] || row.listener).toUpperCase();
  if (replaceExample) {
    const example = EXAMPLE_SENTENCE[sourceLanguage.value];
    if (example && (!sourceText.value.trim() || allExamples.has(sourceText.value.trim()))) {
      sourceText.value = example;
      updateSourceCharacterCount();
    }
  }
}

function selectSource(locale, replaceExample = true) {
  sourceLanguage.value = locale;
  sourcePicker.current.textContent = NATIVE_NAMES[locale] || LANGUAGE_NAMES[locale];
  document.querySelector('#route-source-name').textContent =
    (NATIVE_NAMES[locale] || LANGUAGE_NAMES[locale] || locale).toUpperCase();
  const rows = listenersFor(locale);
  const kept = rows.find(row => row.listener === lastListenerLocale) || rows[0];
  selectListener(kept, false);
  applyGibberishCopy();
  if (replaceExample) {
    const example = EXAMPLE_SENTENCE[locale];
    if (example && (!sourceText.value.trim() || allExamples.has(sourceText.value.trim()))) {
      sourceText.value = example;
      updateSourceCharacterCount();
    }
  }
}

let lastListenerLocale = 'pt-BR';

document.querySelector('#random-listener').addEventListener('click', () => {
  const rows = listenersFor(sourceLanguage.value);
  const weights = rows.map(row => Math.max(row.audibleRules || 0, 1));
  let roll = Math.random() * weights.reduce((a, b) => a + b, 0);
  let pick = rows[0];
  for (let index = 0; index < rows.length; index += 1) {
    roll -= weights[index];
    if (roll <= 0) { pick = rows[index]; break; }
  }
  lastListenerLocale = pick.listener;
  selectListener(pick);
});

sourcePicker.picker.render = () => renderPicker(sourcePicker.picker, {
  items: Object.keys(LISTENERS_BY_SOURCE).map(locale => ({
    locale,
    // A source language counts as curated when any of its directions is.
    curated: listenersFor(locale).some(row => row.curated)
  })),
  selected: sourceLanguage.value,
  onPick: item => { selectSource(item.locale); sourcePicker.close(); sourcePicker.picker.render(); }
});

listenerPicker.picker.render = () => renderPicker(listenerPicker.picker, {
  items: listenersFor(sourceLanguage.value).map(row => ({
    locale: row.listener, curated: row.curated, row
  })),
  selected: lastListenerLocale,
  lead: null,
  onPick: item => {
    lastListenerLocale = item.locale;
    selectListener(item.row);
    listenerPicker.close();
    listenerPicker.picker.render();
  }
});

selectSource('en-US', false);
applyGibberishCopy();

// Reflect candidate availability without blocking the form; the submit path
// still reports the precise service state on any attempt.
fetch('/api/health', { headers: { accept: 'application/json' } })
  .then(response => response.json())
  .then(health => {
    if (!health.azure_lens_enabled) {
      lensStatus.className = 'lens-status notice';
      setLensStatus('Candidate disabled.', 'The Azure listener-lens lane is currently off by flag. Enable AZURE_LENS_CANDIDATE_ENABLED on the Worker to render live pairs.');
    }
  })
  .catch(() => {});

function releaseRuntimeAudio() {
  runtimeUrls.forEach(url => URL.revokeObjectURL(url));
  runtimeUrls = [];
}

function renderPhoneGrid(element, words, phoneFor, changedFor = () => []) {
  element.replaceChildren();
  words.forEach(word => {
    const rules = changedFor(word);
    const cell = document.createElement('div');
    cell.className = `phone-cell${rules.length ? ' changed' : ''}`;
    const written = document.createElement('span');
    written.className = 'written';
    written.textContent = word.written;
    const phone = document.createElement('span');
    phone.className = 'phone';
    phone.textContent = phoneFor(word) || '—';
    cell.title = rules.join(', ');
    cell.append(written, phone);
    element.append(cell);
  });
}

function drillRuleLabel(ruleId) {
  const label = String(ruleId).split('.').at(-1).replaceAll('_', ' ');
  const parts = label.split(' ');
  return parts.length === 2 ? `${parts[0]} → ${parts[1]}` : label;
}

function renderLensPhoneGrids(words, appliedRuleIds) {
  const applied = new Set(appliedRuleIds);
  const activeRules = word => (word.applied_rule_ids || []).filter(id => applied.has(id));
  renderPhoneGrid(document.querySelector('#phone-grid-a'), words, word => word.source_phone);
  renderPhoneGrid(document.querySelector('#phone-grid-b'), words, word => word.lens_phone, activeRules);
}

function renderGibberishPhoneGrids(words) {
  // The source-only lane intentionally has no listener recategorization and
  // therefore no second set of "heard" phones. Keep A readable as the source
  // word sequence and show the generated meaning-opaque phones only on B.
  const firstPhone = word => word.written;
  const secondPhone = word => word.gibberish_phone;
  renderPhoneGrid(document.querySelector('#phone-grid-a'), words, firstPhone);
  renderPhoneGrid(document.querySelector('#phone-grid-b'), words, secondPhone);
}

function setEvidenceList(elementId, values, emptyMessage) {
  const list = document.querySelector(elementId);
  list.replaceChildren();
  const rows = values.length ? values : [emptyMessage];
  rows.forEach(value => {
    const item = document.createElement('li');
    item.textContent = value;
    list.append(item);
  });
}

function renderEvidenceReceipt(payload) {
  const applied = payload.applied_rule_ids || [];
  const context = payload.context_absent_rule_ids || [];
  const neutralized = payload.map_neutralized_rule_ids || [];
  const inaudible = payload.renderer_inaudible_rule_ids || [];
  document.querySelector('#applied-rule-count').textContent = String(applied.length);
  document.querySelector('#context-absent-count').textContent = String(context.length);
  document.querySelector('#neutralized-count').textContent = String(neutralized.length);
  document.querySelector('#drawer-applied-count').textContent = String(applied.length);
  document.querySelector('#drawer-context-count').textContent = String(context.length);
  document.querySelector('#drawer-neutralized-count').textContent = String(neutralized.length);
  document.querySelector('#drawer-inaudible-count').textContent = String(inaudible.length);
  setEvidenceList('#evidence-applied-list', applied, 'No rule was applied.');
  setEvidenceList('#evidence-context-list', context, 'Every implemented rule had matching context.');
  setEvidenceList('#evidence-neutralized-list', neutralized, 'No mapped rule collapsed to the same rendered phone.');
  setEvidenceList('#evidence-inaudible-list', inaudible, 'No matched rule was removed as renderer-inaudible.');
}

function updateDrillSetup(payload) {
  const direction = document.querySelector('#drill-direction');
  const focus = document.querySelector('#drill-focus');
  const sourceName = NATIVE_NAMES[payload.locale] || LANGUAGE_NAMES[payload.locale] || payload.locale;
  const listenerName = NATIVE_NAMES[payload.listener_locale]
    || LANGUAGE_NAMES[payload.listener_locale]
    || payload.listener_locale;
  direction.textContent = `${sourceName} → ${listenerName}`;
  focus.replaceChildren();
  payload.applied_rule_ids.forEach(ruleId => {
    const option = document.createElement('option');
    option.value = ruleId;
    option.textContent = drillRuleLabel(ruleId);
    option.title = ruleId;
    focus.append(option);
  });
  focus.disabled = payload.applied_rule_ids.length === 0;
  activitySubmit.disabled = focus.disabled;
  currentActivityMetadata = focus.disabled ? null : {
    profile_id: payload.profile_id,
    source_locale: payload.locale,
    listener_locale: payload.listener_locale,
    rule_ids: [...payload.applied_rule_ids],
    changed_word_count: payload.affected_word_count,
    comparison_status: 'ready',
    renderer_verification: 'azure_ssml_pair_returned'
  };
}

function resetDrillSetup() {
  const direction = document.querySelector('#drill-direction');
  const focus = document.querySelector('#drill-focus');
  direction.textContent = 'Use the comparison above';
  focus.replaceChildren();
  const option = document.createElement('option');
  option.textContent = 'Generate a listener comparison to load its contrasts';
  focus.append(option);
  focus.disabled = true;
  activitySubmit.disabled = true;
  currentActivityMetadata = null;
}

function appendList(parent, title, values, ordered = false) {
  const heading = document.createElement('h4');
  heading.textContent = title;
  const list = document.createElement(ordered ? 'ol' : 'ul');
  values.forEach(value => {
    const item = document.createElement('li');
    item.textContent = value;
    list.append(item);
  });
  parent.append(heading, list);
}

function renderActivity(payload) {
  const activity = payload.activity;
  const article = document.createElement('article');
  article.className = 'activity-result';
  const source = document.createElement('span');
  source.className = 'result-source';
  source.textContent = payload.source === 'azure_foundry'
    ? `GPT-5.6 · Azure Foundry · ${payload.model}`
    : 'Curated fallback';
  const title = document.createElement('h3');
  title.textContent = activity.title;
  const objective = document.createElement('p');
  objective.textContent = activity.objective;
  article.append(source, title, objective);
  appendList(article, 'Warm-up', activity.warmup);
  appendList(article, 'Listen for', activity.listen_for);
  appendList(
    article,
    'Practice',
    activity.practice_steps.map(step => `${step.minutes} min — ${step.instruction} Teacher note: ${step.teacher_note}`),
    true
  );
  const exitHeading = document.createElement('h4');
  exitHeading.textContent = 'Exit ticket';
  const exit = document.createElement('p');
  exit.textContent = activity.exit_ticket;
  const note = document.createElement('p');
  note.className = 'note';
  note.textContent = activity.evidence_note;
  article.append(exitHeading, exit, note);
  activityOutput.replaceChildren(article);
}

activityForm.addEventListener('submit', async event => {
  event.preventDefault();
  if (!currentActivityMetadata) return;
  activitySubmit.disabled = true;
  activitySubmit.textContent = 'Generating activity…';
  const body = {
    focus: document.querySelector('#drill-focus').value,
    grade_band: document.querySelector('#grade-band').value,
    minutes: Number(document.querySelector('#minutes').value),
    result_metadata: currentActivityMetadata
  };
  try {
    const response = await fetch('/api/activity', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body)
    });
    const payload = await response.json();
    if (!response.ok || !payload.activity) throw new Error('activity_unavailable');
    renderActivity(payload);
  } catch {
    activityOutput.textContent = 'The classroom activity is unavailable right now. Your listening comparison is unchanged.';
  } finally {
    activitySubmit.disabled = false;
    activitySubmit.textContent = 'Generate classroom activity';
  }
});

function renderShiftedSentence(element, words, appliedRuleIds) {
  const appliedSet = new Set(appliedRuleIds);
  element.replaceChildren();
  words.forEach((word, index) => {
    if (index > 0) element.append(document.createTextNode(' '));
    const activeIds = word.applied_rule_ids.filter(id => appliedSet.has(id));
    if (activeIds.length) {
      const mark = document.createElement('mark');
      mark.textContent = word.written;
      mark.title = activeIds.join(', ');
      element.append(mark);
    } else {
      element.append(document.createTextNode(word.written));
    }
  });
}

function azureErrorMessage(error, detail) {
  switch (error) {
    case 'azure_lens_disabled':
      return 'The Azure listener-lens lane is turned off by flag, so no pair was rendered.';
    case 'azure_lens_key_missing':
      return 'The lane is enabled but no Azure Speech key is configured on the service, so no live render ran.';
    case 'azure_lens_rejected':
    case 'unsupported_azure_lens_request': {
      // The service names the word it refused; a refusal the user can act on
      // beats a generic shrug. Only the token is interpolated, via
      // textContent downstream, never markup.
      const token = /token '([^']{1,40})'/.exec(detail || '')?.[1]
        || /in word '([^']{1,40})'/.exec(detail || '')?.[1];
      if (token) {
        return `The word “${token}” uses a sound this lane can't verify. Try writing it differently — numbers and symbols work spelled out as words.`;
      }
      // Two different reasons produce no pair, and the difference is worth
      // telling: nothing matched (try other words) versus rules matched and
      // this voice pronounces both sides identically (the sentence is fine,
      // the renderer is the limit). Only the count is interpolated.
      const collapsed = /(\d+) matched rules? land on sound pairs/.exec(detail || '')?.[1];
      if (collapsed) {
        return `This listener does re-hear ${collapsed === '1' ? 'a sound' : 'sounds'} in that sentence, but the voice pronounces ${collapsed === '1' ? 'that pair' : 'those pairs'} identically, so there was no audible difference to render. Another sentence may reach rules this voice can voice.`;
      }
      return 'That sentence has no supported listener-lens sound shift, so no pair was rendered.';
    }
    case 'azure_render_failed':
    case 'azure_lens_upstream_unreachable':
    case 'azure_lens_upstream_invalid':
    case 'azure_lens_contract_invalid':
      return 'Azure synthesis did not return a valid pair. No unverified audio was substituted.';
    default:
      return 'A gate-passing pair was unavailable. No unverified audio was substituted.';
  }
}

// When a direction has nothing to say about the typed sentence, offer text
// that provably fires: a precomputed word list for the structurally thin
// pairs, else the source language's verified example sentence.
function offerSuggestion() {
  const suggestion = DIRECTION_SUGGESTIONS[listenerProfile.value]
    || (EXAMPLE_SENTENCE[sourceLanguage.value] !== sourceText.value.trim()
        ? EXAMPLE_SENTENCE[sourceLanguage.value] : null);
  if (!suggestion) return;
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'suggestion-chip';
  button.textContent = `Try: “${suggestion}”`;
  button.addEventListener('click', () => {
    sourceText.value = suggestion;
    updateSourceCharacterCount();
    button.remove();
    lensForm.requestSubmit();
  });
  lensStatus.append(document.createTextNode(' '), button);
}

function gibberishErrorMessage(error, detail) {
  switch (error) {
    case 'gibberish_disabled':
      return 'This mode is turned off by flag, so nothing was rendered.';
    case 'gibberish_key_missing':
      return 'The mode is enabled but no Azure Speech key is configured on the service, so no live render ran.';
    case 'gibberish_rejected':
    case 'unsupported_gibberish_request': {
      const token = /token '([^']{1,40})'/.exec(detail || '')?.[1]
        || /gibberish '([^']{1,40})'/.exec(detail || '')?.[1];
      if (token) {
        return `The word “${token}” uses a sound this lane can't verify, so the sentence was refused rather than half-rendered. Try writing it differently — numbers and symbols work spelled out as words.`;
      }
      return 'This source text could not be converted into a complete sound plan. Try ordinary words in the selected language; no partial audio was returned.';
    }
    case 'gibberish_render_failed':
    case 'gibberish_upstream_unreachable':
    case 'gibberish_upstream_invalid':
    case 'gibberish_contract_invalid':
      return 'Azure synthesis did not return a valid pair. No unverified audio was substituted.';
    default:
      return 'A gate-passing pair was unavailable. No unverified audio was substituted.';
  }
}

async function runGibberish(text, signal) {
  const locale = sourceLanguage.value;
  const response = await fetch('/api/gibberish', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text, source_locale: locale }),
    signal
  });
  const payload = await response.json();
  if (!response.ok || payload.status !== 'ready_gibberish_lane') {
    lensStatus.className = 'lens-status notice';
    setLensStatus('Sound-only version unavailable.', gibberishErrorMessage(payload.error, payload.detail));
    return;
  }
  lensResult.hidden = false;
  document.querySelector('#result-source').textContent = payload.normalized_text;
  renderGibberishPhoneGrids(payload.words);
  const syllables = payload.words.reduce((total, word) => total + word.syllable_count, 0);
  document.querySelector('#gibberish-word-count').textContent = String(payload.words.length);
  document.querySelector('#gibberish-syllables').textContent = String(syllables);
  document.querySelector('#gibberish-reduction').textContent =
    payload.vowel_reduction ? 'applied' : 'none';
  const neutralUrl = audioUrl(payload.audio.neutral.wav_base64);
  const gibberishUrl = audioUrl(payload.audio.gibberish.wav_base64);
  runtimeUrls = [neutralUrl, gibberishUrl];
  runtimePlayer.audioUrls = { neutral: neutralUrl, gibberish: gibberishUrl };
  document.querySelector('#runtime-voice-label').textContent =
    `${payload.voice} · ${payload.locale} · same Azure voice on both sides`;
  runtimePlayer.hidden = false;
  playerControllers.get(runtimePlayer).choose(runtimePlayer.querySelector('[data-side="neutral"]'));
  paintPlayerWaveforms(runtimePlayer, {
    neutral: payload.audio.neutral.wav_base64,
    gibberish: payload.audio.gibberish.wav_base64
  });
  lensStatus.className = 'lens-status success';
  const languageName = LANGUAGE_NAMES[payload.locale] || payload.locale;
  const built =
    `${payload.words.length} word${payload.words.length === 1 ? '' : 's'} rebuilt from ${languageName}'s ${payload.core_size} most common syllables, keeping ${syllables} syllable${syllables === 1 ? '' : 's'} in the same places. ` +
    (payload.vowel_reduction
      ? 'Unstressed vowels reduce toward schwa, as they do in this language. '
      : 'No vowel reduction: this language keeps its unstressed vowels full. ');
  const calls = `Azure calls: ${payload.api_calls_made}${payload.cache_hit ? ' · served from cache' : ''}.`;
  setLensStatus('Sound-only version ready.',
    `${built}The same sentence always produces the same nonsense. ${calls}`);
}

lensForm.addEventListener('submit', async event => {
  event.preventDefault();
  const text = sourceText.value;
  const profileId = listenerProfile.value;
  lensSubmit.disabled = true;
  lensSubmit.textContent = 'Rendering…';
  lensStatus.className = 'lens-status working';
  setLensStatus('Working.', mode === 'gibberish'
    ? 'Rebuilding the sentence from this language’s own syllables, then rendering both sides through Azure SSML.'
    : 'Rendering the neutral and listener-lens pair through Azure SSML.');
  lensResult.hidden = true;
  runtimePlayer.hidden = true;
  resetDrillSetup();
  releaseRuntimeAudio();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    if (mode === 'gibberish') {
      await runGibberish(text, controller.signal);
      return;
    }
    const response = await fetch('/api/azure-lens', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text, profile_id: profileId }),
      signal: controller.signal
    });
    const payload = await response.json();
    if (!response.ok || payload.status !== 'ready_azure_lane') {
      lensStatus.className = 'lens-status notice';
      setLensStatus('Comparison unavailable.', azureErrorMessage(payload.error, payload.detail));
      if (payload.error === 'azure_lens_rejected' && !/token '|in word '/.test(payload.detail || '')) {
        offerSuggestion();
      }
      return;
    }
    lensResult.hidden = false;
    renderShiftedSentence(document.querySelector('#result-source'), payload.words, payload.applied_rule_ids);
    renderLensPhoneGrids(payload.words, payload.applied_rule_ids);
    renderEvidenceReceipt(payload);
    updateDrillSetup(payload);
    document.querySelector('#rule-count').textContent = String(payload.affected_word_count);
    document.querySelector('#active-rule').textContent =
      payload.applied_rule_ids.length ? payload.applied_rule_ids.join(', ') : '—';
    const neutralUrl = audioUrl(payload.audio.neutral.wav_base64);
    const lensUrl = audioUrl(payload.audio.lens.wav_base64);
    const speakerUrl = audioUrl(payload.audio.speaker.wav_base64);
    runtimeUrls = [neutralUrl, lensUrl, speakerUrl];
    runtimePlayer.audioUrls = { neutral: neutralUrl, lens: lensUrl, speaker: speakerUrl };
    document.querySelector('#runtime-voice-label').textContent =
      `A/B: ${payload.voice} (one voice, only the ear's edits differ) · C: ${payload.speaker_voice}`;
    runtimePlayer.hidden = false;
    playerControllers.get(runtimePlayer).choose(runtimePlayer.querySelector('[data-side="neutral"]'));
    paintPlayerWaveforms(runtimePlayer, {
      neutral: payload.audio.neutral.wav_base64,
      lens: payload.audio.lens.wav_base64,
      speaker: payload.audio.speaker.wav_base64
    });
    lensStatus.className = 'lens-status success';
    const ruleCount = payload.applied_rule_ids.length;
    if (ruleCount === 0) queueMicrotask(offerSuggestion);
    const contextAbsent = (payload.context_absent_rule_ids || []).length;
    const inaudible = (payload.renderer_inaudible_rule_ids || []).length;
    const subtle = ruleCount >= 1 && ruleCount <= 2;
    const notes = [
      `${payload.affected_word_count} word${payload.affected_word_count === 1 ? '' : 's'} changed with ${ruleCount} rule${ruleCount === 1 ? '' : 's'}.`,
      contextAbsent ? `${contextAbsent} did not match this sentence.` : 'Every available rule found matching context.',
      inaudible ? `${inaudible} matched change${inaudible === 1 ? ' was' : 's were'} inaudible in this voice.` : '',
      payload.prosody?.contour_applied ? 'Question contour applied.' : '',
      payload.cache_hit ? 'Cached result.' : `${payload.api_calls_made} audio renders.`
    ].filter(Boolean);
    setLensStatus(subtle ? 'Comparison ready — subtle changes.' : 'Comparison ready.', notes.join(' '));
  } catch {
    lensStatus.className = 'lens-status notice';
    setLensStatus('Comparison unavailable.', 'The service did not return in time. Try again.');
  } finally {
    clearTimeout(timeout);
    lensSubmit.disabled = false;
    lensSubmit.textContent = submitLabel();
  }
});
