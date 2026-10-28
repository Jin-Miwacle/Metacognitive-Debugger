/* CONFIG */
/* The backend address lives in config.js and nowhere else, so there is only one
   line to change when the tunnel restarts. CHECKIN_EVERY_SEC is the only default
   kept here, since it is a behaviour setting rather than an address. */
const CONFIG = Object.assign({ API_BASE: '', CHECKIN_EVERY_SEC: 90 }, window.APP_CONFIG || {});
if (!CONFIG.API_BASE) console.warn('No API_BASE in config.js, so the app will use its offline replies.');

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

/* Built-in practice readings */
const DEMO_READINGS = [
  {
    id: 'demo-lamp',
    title: 'The Lamp on Harrow Point',
    text:
`Every evening at dusk, Mara climbed the ninety-one steps of the lighthouse to light the lamp, just as her grandfather had done before her. The lamp itself was old, but it never failed. What worried her was the harbor below, where fewer boats returned each season.

Tonight, a single fishing boat wobbled toward the rocks. Mara did not wave or shout. She turned the lamp's shutter slowly, three long flashes and one short, the way her grandfather had taught her when she was small. The boat straightened and slid into the channel between the stones.

At the dock, the fisherman looked up at the tower and lifted his cap. He had never met Mara, yet he knew exactly who was up there. The old signal had told him that the person keeping the light still remembered the old ways, and that the channel was safe to trust.`
  },
  {
    id: 'demo-seed',
    title: 'The Last Jar of Seeds',
    text:
`When the rains failed for the second year, the community garden shrank to a single raised bed behind the old schoolhouse. Teodora kept one jar of bean seeds on the highest shelf in her kitchen, untouched, while the rest of the garden's seeds went into the ground early, before anyone was sure the soil could hold them.

Her neighbors asked why she didn't plant her share like everyone else. She only said that a garden needed one thing kept in reserve, in case the first planting failed entirely. When a late frost killed half the seedlings in May, it was her jar, and no one else's, that refilled the empty rows by the second week of June.

Nobody suggested she explain herself again after that. The following spring, three other families on her street had jars of their own on their highest shelves, untouched, waiting.`
  },
  {
    id: 'demo-machine',
    title: 'The Quiet Machine',
    text:
`The record player had sat in the back room of the library for eleven years, donated along with a box of records nobody had catalogued. When Priya finally opened the cabinet to clear space, she found a needle worn down to almost nothing and a handwritten label taped inside the lid: "Return to Mr. Oyelaran when fixed."

She looked him up before she decided what to do with any of it. Mr. Oyelaran had taught music at the school across the street until it closed in the nineties; the library had no record of him after that. Still, she ordered a new needle, cleaned the turntable herself over a weekend, and played the first record in the box just to test it.

It was a recording of a school choir, slightly out of tune, dated the same year the label was written. Priya didn't know whose voice was whose. She reshelved the player in the front room anyway, with a handwritten card of her own: "Ask at the desk to listen."`
  },
  {
    id: 'demo-noli',
    title: 'Kabanata 1',
    text:
`Isang marangyang salu-salo ang ipinag-anyaya ni Don Santiago de los Santos na higit na popular sa taguring Kapitan Tiago.  Ang handaan ay gagawin sa kanyang bahay na nasa daang Anluwage na karatig ng Ilog-Binundok.

Ang paayaya ay madaling kumalat sa lahat ng sulok ng Maynila.  Bawat isa ay gustong dumalo sapagkat ang mayamang Kapitan ay kilala bilang isang mabuting tao, mapagbigay at laging bukas ang palad sa mga nangangailangan.  Dahil dito, ang iba ay nababalino kung ano ang isusuot at sasabihin sa mismong araw ng handaan.

Nang gabing iyon dagsa ang mga panauhin na gaya ng dapat asahan.  Puno ang bulwagan.  Ang nag-iistima sa mgta bisita ay si Tiya Isabel, isang matandang babae na pinsan ng may-bahay.  Kabilang sa mga bisita sina Tinyente ng guardia civil, Pari Sibyla, ang Kura paroko ng Binundok, si Padre Damaso na madaldal at mahahayap ang mga salita at dalawang Paisano.  Ang isa ay kararating lamang sa Pilipinas.`
  }
];
const DEMO_READING_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 3.6 4.8 7.4l7.2 3.8 7.2-3.8Z"/><path d="M4.8 7.4v9l7.2 3.8 7.2-3.8v-9"/><path d="M12 11.2v9"/></svg>`;

/* ---------- State for the reading workspace (Goal / Debug / Check-in / Progress) ---------- */
const state = {
  goal: null,            // { text, confidence }
  threads: [],           // debug sessions
  pending: null,         // current text selection { text, paragraph, range }
  selectionNote: '',
  checkin: {
    on: false, everySec: CONFIG.CHECKIN_EVERY_SEC, nextAt: 0, pending: null, i: 0, responses: [], fetching: false, shown: [],
    lastTick: Date.now(), held: false, popupHidden: false,
    // Whether a check-in shows as a floating popup the moment it's ready, instead of
    popupOn: (() => { try { return localStorage.getItem('checkinPopupOn') !== 'off'; } catch (e) { return true; } })()
  },
  support: { score: 0, level: 'steady' },  // see nudgeSupport() below
  responseLanguage: (() => {              // 'auto' | 'en' | 'fil' -- which language the coach replies in
    try { return localStorage.getItem('responseLanguage') || 'auto'; } catch (e) { return 'auto'; }
  })(),
  startedAt: Date.now(),
  log: [],
  user: null,                  // { id, role, full_name } once logged in
  reading: { id: null },
  pastedTitle: null,           // what the student called their own text, shown in History
  session: { id: null }        // the row in the "sessions" table for this reading attempt
};
function logEvent(type, data = {}) {
  state.log.push({ t: new Date().toISOString(), type, ...data });
}

/* ---- Support level: makes goals/check-ins actually change the session ---- */
function nudgeSupport(delta) {
  const s = state.support;
  s.score = Math.max(0, Math.min(6, s.score + delta));
  const prevLevel = s.level;
  s.level = s.score >= 4 ? 'extra' : s.score >= 2 ? 'closer' : 'steady';
  if (s.level !== prevLevel) logEvent('support_level_change', { from: prevLevel, to: s.level, score: s.score });
  updateSupportBanner();
  if (currentTab() === 'monitor') renderCheckin();
  if (currentTab() === 'stats') renderStats();
}
function effectiveCheckinSec() {
  const base = state.checkin.everySec;
  if (state.support.level === 'extra') return Math.max(30, Math.round(base * 0.5));
  if (state.support.level === 'closer') return Math.max(30, Math.round(base * 0.75));
  return base;
}
function updateSupportBanner() {
  const el = $('#supportBanner'); if (!el) return;
  const s = state.support;
  if (s.level === 'steady') { el.hidden = true; return; }
  el.hidden = false;
  el.textContent = s.level === 'extra'
    ? 'Extra support: check-ins are more frequent while we work through this together.'
    : 'Checking in a little more often: a few spots have been tricky.';
}
/* Quick word-overlap check against the goal (stopwords stripped) */
const STOPWORDS = new Set(['the','a','an','to','of','in','on','for','and','or','is','are','was','were','it','its','that','this','what','why','how','do','does','did','i','me','my','understand','about','with','as','be','at','so','not']);
function keywordsOf(text) {
  return new Set((text.toLowerCase().match(/[a-z']+/g) || []).filter(w => w.length > 2 && !STOPWORDS.has(w)));
}
function goalOverlapWord(passage) {
  if (!state.goal) return null;
  const gw = keywordsOf(state.goal.text), pw = keywordsOf(passage);
  for (const w of pw) if (gw.has(w)) return w;
  return null;
}

/* ---- Saving progress to the database ---- */

/* What has failed to save this session */
const saveHealth = { failures: 0, offline: false, last: '' };

function renderSaveWarning() {
  let box = $('#saveWarning');
  if (!box) {
    box = document.createElement('div');
    box.id = 'saveWarning';
    box.className = 'save-warning';
    // The sidebar is on every signed-in view; fall back to the body so the
    ($('.sidebar-foot') || document.body).prepend(box);
  }
  const n = saveHealth.failures;
  // The bold heading and the count ran together as "Not saving1 write failed."
  box.innerHTML = `<b>Not saving.</b> <span>${n} write${n === 1 ? '' : 's'} failed. `
    + `Your work is on screen but not in the database.</span>`;
  box.title = saveHealth.last;
}

/* True when the database refused a write because a column a migration adds is not there yet, as */
function isMissingColumn(error, column) {
  const m = (error && error.message) || '';
  return new RegExp("column .*" + column + ".* does not exist|'" + column + "' column|schema cache", 'i').test(m);
}
function noteSaveFailure(what, error) {
  const message = (error && error.message) || String(error || 'unknown error');
  console.warn('Could not save ' + what + ':', message);
  saveHealth.failures += 1;
  saveHealth.last = what + ': ' + message;
  if (saveHealth.failures === 1 && typeof showToast === 'function') {
    showToast('Your work is not being saved. Tell your teacher before you carry on.', 'bug');
  }
  renderSaveWarning();
}

/* The offline fallback is its own kind of quiet: the app keeps answering, but from the built-in demo */
function noteOfflineFallback() {
  if (saveHealth.offline) return;
  saveHealth.offline = true;
  if (typeof showToast === 'function') {
    showToast('The backend could not be reached, so replies are coming from the built-in offline examples.', 'bug');
  }
}
async function ensureSession() {
  if (state.session.id) return state.session.id;
  if (!state.user) return null;
  const row = {
    student_id: state.user.id,
    reading_id: state.reading.id,
    goal: state.goal ? state.goal.text : null,
    confidence: state.goal ? state.goal.confidence : null
  };
  // Only pasted text carries a title of its own. Sent separately so the
  if (!state.reading.id && state.pastedTitle) row.title = state.pastedTitle;

  let { data, error } = await supa.from('sessions').insert(row).select('id').single();
  if (error && row.title && isMissingColumn(error, 'title')) {
    // session_title_migration.sql has not been run. Losing the title is a
    delete row.title;
    ({ data, error } = await supa.from('sessions').insert(row).select('id').single());
    console.warn('Saved this session without its title. Run session_title_migration.sql to keep names for pasted texts.');
  }
  if (error) { noteSaveFailure('the reading session', error); return null; }
  state.session.id = data.id;
  return data.id;
}
async function dbUpdateSession(fields) {
  const id = state.session.id; if (!id) return;
  const { error } = await supa.from('sessions').update(fields).eq('id', id);
  if (error) noteSaveFailure('the session update', error);
}
async function dbInsertDebugEvent(t) {
  const sid = await ensureSession(); if (!sid) return;
  const { data, error } = await supa.from('debug_events')
    .insert({ session_id: sid, passage: t.text, problem: t.problem,
              problem_text: t.problemText || null, status: 'open' })
    .select('id').single();
  if (error) { noteSaveFailure('this flagged passage', error); return; }
  t.dbId = data.id;
}
async function dbUpdateDebugStatus(t) {
  if (!t.dbId) return;
  const { error } = await supa.from('debug_events').update({ status: t.status }).eq('id', t.dbId);
  if (error) noteSaveFailure('the change to this flagged passage', error);
}
/* Saves the actual back-and-forth for this debug thread */
async function dbSaveTranscript(t) {
  if (!t.dbId) return;
  const transcript = t.msgs.filter(m => !m.typing).map(m => ({ from: m.from, text: m.text, demo: !!m.demo }));
  const { error } = await supa.from('debug_events').update({ transcript }).eq('id', t.dbId);
  if (error) noteSaveFailure('the conversation about this passage', error);
}
async function dbInsertCheckin(prompt, level, answer) {
  const sid = await ensureSession(); if (!sid) return;
  const { error } = await supa.from('checkin_events').insert({ session_id: sid, prompt, level, answer });
  if (error) noteSaveFailure('this check-in answer', error);
}

/* ---------- Offline demo quiz (used only if the backend is unavailable, and only for the demo story) ---------- */
const OFFLINE_QUIZ = [
  { kind: 'Literal', q: 'How many steps does Mara climb to reach the lamp?',
    options: ['Nineteen', 'Ninety-one', 'Ninety-nine', 'One hundred nine'], answer: 1,
    why: 'The first sentence states the number directly.' },
  { kind: 'Inference', q: 'Why does Mara use the lamp shutter instead of waving or shouting?',
    options: ['She is afraid of the fisherman', 'The lamp is broken', 'The flashes are a known signal that guides boats through the channel', 'She wants to save oil'], answer: 2,
    why: 'The text never says "it was a code", but the boat straightens and enters the channel right after the pattern, which tells us the signal carried meaning.' },
  { kind: 'Inference', q: 'What does "he knew exactly who was up there" suggest?',
    options: ['Mara told him in a letter', 'He recognized the keepers’ traditional signal', 'He could see her face from the boat', 'He was her grandfather’s friend'], answer: 1,
    why: 'He had never met her, so he must have known her from something else: the old signal that the keepers use.' }
];

/* ---------- The four kinds of trouble a student can pick ---------- */
const PROBLEMS = {
  vocab:   { label: 'A word I don’t know', sub: 'Unfamiliar or tricky word' },
  lost:    { label: 'I lost the point',       sub: 'I read it but it didn’t land' },
  connect: { label: 'Ideas don’t connect', sub: 'How does this link to before?' },
  why:     { label: 'Why did this happen?',   sub: 'Something is implied, not stated' }
};
/* A student whose trouble is none of the four types above types it into the box under them instead */

/* ---------- Offline demo replies (used ONLY if the backend can't be reached) ---------- */
const DEMO_REPLIES = {
  vocab: [
    'Pick the one word that blocks you. Reread the sentence with that word skipped. Do you still get the gist? Then look at the sentences around it for clues: an example, a contrast, or a restatement.',
    'Which word is it? Try to guess its meaning from the sentence around it. What would make sense there? Then check your guess against the next sentence.'
  ],
  lost: [
    'Reread just this sentence once, slowly. Then say it in your own words in ten words or fewer. Who is doing what?',
    'Try this: what is this sentence adding that the previous one did not say? Say the main idea out loud before you move on.'
  ],
  connect: [
    'Look at the sentence right before this one. How are they linked: one causes the other, they contrast, or this adds more? Try putting “because”, “but”, or “and then” between them. Which fits?',
    'Find the last thing you clearly understood. What changed between that point and this one? Name the link in one short phrase.'
  ],
  why: [
    'Split it in two. What does the text say directly, and what are you filling in yourself? Which earlier detail supports your guess?',
    'Ask: what would have to be true for this to make sense? Look back for a clue that points that way.'
  ],
  other: [
    'Say the trouble in one sentence, as plainly as you can. Then reread just this part and watch for the exact word or idea it sits on.',
    'Start from what you do follow here. Then name the first thing that stops making sense. That spot is usually the one worth working on.'
  ],
  followup: [
    'Compare your version with the original: did you keep who did what? If something is missing, add it. Then tell me below if it makes sense now.',
    'Check one thing: does your explanation still hold when you read the next sentence? If yes, mark it as clear.'
  ],
  stuck: [
    'That’s okay. This one is flagged so your teacher can see it. Break it down: what is the subject, what is it doing, and which earlier line does it depend on?'
  ],
  // Shown after a student has struggled through a couple of follow-ups in a
  giveup: [
    'No worries -- here is the key idea plainly: focus on who is doing what in this sentence, and what changed right before it. Reread it once more with that in mind, then tell me if it clicks.',
    'That is okay, let us just walk through it directly: look at the subject of the sentence and the one earlier detail it depends on. Put those two together and see if it makes sense now.'
  ]
};
const pick = arr => arr[Math.floor(Math.random() * arr.length)];

/* ---------- Talking to the backend ---------- */
async function callBackend(path, body) {
  if (!CONFIG.API_BASE) return null;
  try {
    const r = await fetch(CONFIG.API_BASE.replace(/\/$/, '') + path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
    if (r.ok) return await r.json();
    console.warn('The backend answered with an error:', r.status, path);
  } catch (e) {
    // A dead tunnel and a CORS misconfiguration look identical from here: neither
    // sends a reply the browser will let us read. Name both, so the console line
    // is worth something to whoever reads it.
    console.warn('No reply from the backend at ' + CONFIG.API_BASE + path
      + ', so the offline demo is answering instead. Open ' + CONFIG.API_BASE
      + ' in a tab: {"status":"ok"} means this is CORS and the backend needs'
      + ' ALLOWED_ORIGINS=' + location.origin
      + '; a Cloudflare error page means the address in config.js is out of date.', e);
    noteOfflineFallback();
  }
  return null;
}

/* The backend reports which kind of turn it decided this was */
async function askPartner(payload) {
  const d = await callBackend('/partner', payload);
  if (d && d.reply) return { reply: d.reply, mode: d.mode || null, demo: false };
  await new Promise(r => setTimeout(r, 450));
  let reply;
  if (payload.mode === 'debug') reply = pick(DEMO_REPLIES[payload.problem] || DEMO_REPLIES.lost);
  else if (payload.mode === 'giveup' || payload.mode === 'explain') reply = pick(DEMO_REPLIES.giveup);
  else if (payload.mode === 'stuck') reply = pick(DEMO_REPLIES.stuck);
  else reply = pick(DEMO_REPLIES.followup);
  // Offline replies carry the mode the frontend asked for, so the ladder
  return { reply, mode: payload.mode || null, demo: true };
}

function partnerPayload(mode, t, message) {
  return {
    mode, problem: t.problem, problem_text: t.problemText || null,
    passage: t.text, paragraph: t.paragraph,
    // the whole reading, so the backend's grounding rules can check names and
    full_text: reader.innerText.trim().slice(0, 20000),
    goal: state.goal && state.goal.text, goal_hint: t.goalHint || null,
    support: state.support.level, message: message || null,
    // 'auto' means let the backend detect the passage's language itself;
    response_language: state.responseLanguage === 'auto' ? null : state.responseLanguage,
    history: t.msgs.filter(m => !m.typing).map(m => ({
      role: m.from === 'you' ? 'student' : 'coach',
      text: m.text,
      // the backend counts coach turns with mode "stuck" to decide how
      mode: m.mode || null
    }))
  };
}

function setResponseLanguage(lang) {
  state.responseLanguage = lang;
  try { localStorage.setItem('responseLanguage', lang); } catch (e) { /* private browsing etc -- still works for this session */ }
  renderStats();
}

/* ---- Offline check-in prompts: backend-unavailable fallback ---- */
const OFFLINE_CHECKINS = [
  'Pause for a moment. In one sentence, what have you read so far?',
  'Is what you just read matching your goal? What is still missing?',
  'What is the most confusing part so far? Try selecting it in the text.',
  'What do you think happens or is explained next?',
  'Which sentence would you reread if you had to explain this to a friend?'
];

/* ---------- Elements ---------- */
const reader = $('#reader');

/* ---------- Side-panel tabs (Goal / Debug / Check-in / Progress) ---------- */
function setTab(name) {
  $$('.tab').forEach(t => t.setAttribute('aria-selected', String(t.dataset.tab === name)));
  $$('.panel').forEach(p => p.hidden = p.id !== 'p-' + name);
  if (name === 'stats') renderStats();
  if (name === 'monitor') renderCheckin();
}
$$('.tab').forEach(t => t.addEventListener('click', () => setTab(t.dataset.tab)));
function currentTab() { return ($$('.tab').find(t => t.getAttribute('aria-selected') === 'true') || {}).dataset?.tab; }

function updateBadges() {
  const has = !!state.checkin.pending;
  $('#monDot').hidden = !(has && currentTab() !== 'monitor');
}

/* ---- Goal tab ---- */
function goalFormHtml(g) {
  return `
    <label class="f" for="goalText">What do you want to get out of this text?</label>
    <textarea id="goalText" placeholder="Example: Understand why Mara uses the lamp signal.">${g ? esc(g.text) : ''}</textarea>
    <label class="f">How well do you expect to understand it?</label>
    <div class="scale" id="conf">
      ${[1,2,3,4,5].map(n => `<button type="button" data-n="${n}" aria-pressed="${g && g.confidence === n}">${n}</button>`).join('')}
    </div>
    <div class="scale-cap"><span>Not at all</span><span>Very well</span></div>`;
}
/* Shared by the Goal tab and the start-of-session popup, so the two never drift out of sync */
function wireGoalForm(scope, g, onDone) {
  let conf = g ? g.confidence : null;
  $$('#conf button', scope).forEach(b => b.addEventListener('click', () => {
    conf = +b.dataset.n; $$('#conf button', scope).forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  }));
  return () => {
    const text = $('#goalText', scope).value.trim();
    if (!text) { $('#goalText', scope).focus(); return false; }
    state.goal = { text, confidence: conf };
    logEvent('goal_set', { goal: text, confidence: conf });
    $('#goalLine').textContent = 'Goal: ' + text;
    if (state.session.id) dbUpdateSession({ goal: text, confidence: conf });
    else ensureSession();
    // Low confidence starts check-ins closer together; high confidence starts
    nudgeSupport({1:2, 2:1, 3:0, 4:-1, 5:-1}[conf ?? 3]);
    // Check-ins already started with the reading. Confidence only moves the next one.
    if (state.checkin.on) state.checkin.nextAt = Date.now() + effectiveCheckinSec() * 1000;
    if (onDone) onDone();
    return true;
  };
}
function renderGoal() {
  const g = state.goal;
  const el = $('#p-goal');
  el.innerHTML = `
    <h3>Before you read</h3>
    <p class="lead">Set a purpose. Readers who know what they want from a text notice sooner when they lose it.</p>
    ${goalFormHtml(g)}
    <div class="row" style="margin-top:14px">
      <button class="btn" id="setGoal" type="button">${g ? 'Update goal' : 'Set goal and start reading'}</button>
    </div>
    ${g ? '<p class="small" style="margin-top:10px">Goal saved. Check-ins are on. Change them in the Check-in tab.</p>' : ''}`;
  const save = wireGoalForm(el, g, () => { renderGoal(); renderDebug(); setTab('debug'); });
  $('#setGoal', el).addEventListener('click', save);
}

/* ---- Goal popup: shown once per session the moment a reading opens, so setting a purpose isn't */
function ensureGoalDialog() {
  let dlg = $('#goalDlg');
  if (dlg) return dlg;
  dlg = document.createElement('dialog');
  dlg.id = 'goalDlg';
  document.body.appendChild(dlg);
  return dlg;
}
function showGoalPopup() {
  const dlg = ensureGoalDialog();
  dlg.innerHTML = `
    <form method="dialog">
      <h3 style="margin:0 0 4px">Before you start reading</h3>
      <p class="small" style="margin:0 0 14px">Set a purpose for this text -- it helps you notice sooner if you lose the thread.</p>
      ${goalFormHtml(null)}
      <div class="row" style="margin-top:14px; justify-content:flex-end">
        <button class="btn alt" type="button" id="goalSkip">Skip for now</button>
        <button class="btn" type="button" id="goalSave">Start reading</button>
      </div>
    </form>`;
  const save = wireGoalForm(dlg, null, () => { dlg.close(); renderGoal(); renderDebug(); });
  $('#goalSave', dlg).addEventListener('click', () => { if (save()) dlg.close(); });
  $('#goalSkip', dlg).addEventListener('click', () => dlg.close());
  dlg.showModal();
}

/* ---------- Text selection -> Debug ---------- */
let selTimer = null, pointerDown = false;
document.addEventListener('selectionchange', () => {
  clearTimeout(selTimer);
  selTimer = setTimeout(captureSelection, 350);
});
document.addEventListener('pointerdown', () => { pointerDown = true; });
document.addEventListener('pointerup', () => {
  pointerDown = false;
  clearTimeout(selTimer);
  selTimer = setTimeout(captureSelection, 50);
});
function captureSelection() {
  if (pointerDown) return;
  const sel = getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) return;
  const range = sel.getRangeAt(0);
  if (!reader.contains(range.commonAncestorContainer)) return;
  if (snapToWords(range)) { sel.removeAllRanges(); sel.addRange(range); return; }
  const text = sel.toString().replace(/\s+/g, ' ').trim();
  if (text.length < 2) return;
  const pA = closestP(range.startContainer), pB = closestP(range.endContainer);
  state.selectionNote = pA !== pB ? 'Select inside a single paragraph so the mark stays clean.' : '';
  state.pending = pA === pB ? { text, paragraph: pA ? pA.textContent : '', range: range.cloneRange() } : null;
  setTab('debug');
  renderDebug();
}
/* Moves the range's ends out to word boundaries and in past any spaces at the edges */
const WORD_CH = /[\p{L}\p{N}\p{M}]/u, JOIN_CH = /['’\-]/;
function snapToWords(range) {
  const before = [range.startContainer, range.startOffset, range.endContainer, range.endOffset];
  const inWord = (s, i) => i >= 0 && i < s.length &&
    (WORD_CH.test(s[i]) || (JOIN_CH.test(s[i]) && WORD_CH.test(s[i - 1] || '') && WORD_CH.test(s[i + 1] || '')));
  const sNode = range.startContainer, eNode = range.endContainer;
  if (sNode.nodeType === 3) {
    const s = sNode.data; let i = range.startOffset;
    while (i < s.length && /\s/.test(s[i]) && !(sNode === eNode && i >= range.endOffset)) i++;
    while (inWord(s, i - 1) && inWord(s, i)) i--;
    range.setStart(sNode, i);
  }
  if (eNode.nodeType === 3) {
    const s = eNode.data; let j = range.endOffset;
    while (j > 0 && /\s/.test(s[j - 1]) && !(sNode === eNode && j <= range.startOffset)) j--;
    while (inWord(s, j) && inWord(s, j - 1)) j++;
    range.setEnd(eNode, j);
  }
  const after = [range.startContainer, range.startOffset, range.endContainer, range.endOffset];
  return before.some((v, k) => v !== after[k]);
}
function closestP(node) {
  const el = node.nodeType === 1 ? node : node.parentElement;
  return el ? el.closest('#reader p') : null;
}

/* ---------- Debug tab ---------- */
function shorten(s, n = 90) { return s.length > n ? s.slice(0, n - 1) + '…' : s; }

/* What the tag on an open thread says */
function threadProblemLabel(t) {
  if (t.problem === 'other' && t.problemText) return shorten(t.problemText, 48);
  return (PROBLEMS[t.problem] || PROBLEMS.lost).label;
}

function renderDebug() {
  const el = $('#p-debug');
  rememberDrafts(el);
  const p = state.pending;
  const goalWord = p ? goalOverlapWord(p.text) : null;
  const chooser = p ? `
    <div class="card">
      <p class="small" style="margin:0 0 4px">You selected</p>
      <blockquote class="quote" style="margin:0 0 10px; padding-left:10px; border-left:3px solid var(--accent); font-family:var(--serif); font-size:15px">${esc(shorten(p.text, 160))}</blockquote>
      ${goalWord ? `<p class="small" style="margin:0 0 8px; color:var(--accent)">This looks related to your goal (“${esc(goalWord)}”), worth sorting out before moving on.</p>` : ''}
      <p class="small" style="margin:0 0 8px">What is going wrong?</p>
      <div class="chips">
        ${Object.entries(PROBLEMS).map(([k, v]) => `<button type="button" data-problem="${k}">${v.label}<small>${v.sub}</small></button>`).join('')}
      </div>
      <form id="ownProblemForm" style="margin-top:12px">
        <label class="f" for="ownProblem">Or say it in your own words</label>
        <div class="row">
          <input type="text" id="ownProblem" style="flex:1" placeholder="What is confusing about it?" autocomplete="off">
          <button class="btn" type="submit">Start</button>
        </div>
      </form>
      <div class="row" style="margin-top:10px"><button class="btn alt" id="cancelSel" type="button">Cancel</button></div>
    </div>` : `
    <div class="card">
      <b>Debug a confusing spot</b>
      <p class="small" style="margin:4px 0 0">${state.selectionNote ? esc(state.selectionNote) : 'Select a word, phrase, or sentence in the text. It gets a red squiggle and we work it out together.'}</p>
    </div>
    <div class="card" style="margin-top:12px">
      <b>Or ask about the reading</b>
      <p class="small" style="margin:4px 0 10px">For something that is not in one spot, like what is going on or why someone did something.</p>
      <form id="askAnyForm">
        <div class="row">
          <input type="text" id="askAny" style="flex:1" placeholder="What do you want to ask?" autocomplete="off">
          <button class="btn" type="submit">Ask</button>
        </div>
      </form>
    </div>`;
  const threads = state.threads.slice().reverse().map(t => `
    <div class="card ${t.status}" data-id="${t.id}">
      <span class="tag ${t.status}">${t.status === 'resolved' ? 'Clear now' : t.status === 'stuck' ? 'Still stuck' : esc(threadProblemLabel(t))}</span>
      ${t.text ? `<blockquote class="quote">${esc(shorten(t.text, 140))}</blockquote>` : ''}
      ${t.msgs.map(m => `<div class="msg ${m.from}${m.typing ? ' typing' : ''}">${esc(m.text)}${m.demo ? '<div class="small" style="margin-top:4px">Offline demo reply (backend not reached)</div>' : ''}</div>`).join('')}
      ${t.status !== 'resolved' && !t.busy ? `
        <form class="reply" data-id="${t.id}">
          <input type="text" placeholder="Explain it in your own words" aria-label="Your reply" autocomplete="off">
          <button class="btn" type="submit">Send</button>
        </form>
        <div class="row" style="margin-top:8px">
          <button class="btn alt" type="button" data-act="resolve" data-id="${t.id}">Makes sense now</button>
          ${t.gaveUp ? '' : `<button class="btn alt" type="button" data-act="stuck" data-id="${t.id}">Still stuck</button>`}
        </div>
        ${coachHasReplied(t) ? `
        <div class="row" style="margin-top:6px">
          <button class="btn alt" type="button" data-act="explain" data-id="${t.id}">${t.explained ? "I still don't get it" : 'Explain it to me'}</button>
        </div>` : ''}` : ''}
      ${t.status === 'resolved' && t.autoResolved ? `
        <div class="row" style="margin-top:8px; align-items:center">
          <span class="small">Marked as cleared up, since your answer showed you had it.</span>
          <button class="btn alt" type="button" data-act="reopen" data-id="${t.id}">Still confused</button>
        </div>` : ''}
    </div>`).join('');
  el.innerHTML = chooser + threads;

  $$('.chips button', el).forEach(b => b.addEventListener('click', () => startDebug(b.dataset.problem)));
  const c = $('#cancelSel', el); if (c) c.addEventListener('click', () => { state.pending = null; getSelection().removeAllRanges(); renderDebug(); });
  const af = $('#askAnyForm', el);
  if (af) af.addEventListener('submit', e => {
    e.preventDefault();
    const box = $('#askAny', el);
    const q = box.value.trim();
    if (!q) return;
    box.value = '';
    drafts.ask = '';
    askAboutReading(q);
  });

  restoreDrafts(el);
}

/* Half-typed text, kept across re-renders. The panel redraws whenever any
   thread changes, and a reply arriving in one thread used to wipe whatever the
   student was typing in another, or in the ask box. */
const drafts = { ask: '', own: '', reply: {} };

/* Blanks the live inputs as well as the held copies. Clearing only the held
   copies does not work: the next render reads the old inputs first and puts
   the text straight back. */
function clearDrafts() {
  drafts.ask = ''; drafts.own = ''; drafts.reply = {};
  const el = $('#p-debug');
  if (!el) return;
  const ask = $('#askAny', el); if (ask) ask.value = '';
  const own = $('#ownProblem', el); if (own) own.value = '';
  $$('form.reply input', el).forEach(i => { i.value = ''; });
}

function rememberDrafts(el) {
  if (!el) return;
  const ask = $('#askAny', el); if (ask) drafts.ask = ask.value;
  const own = $('#ownProblem', el); if (own) drafts.own = own.value;
  $$('form.reply', el).forEach(f => {
    const i = f.querySelector('input');
    if (i) drafts.reply[f.dataset.id] = i.value;
  });
}

function restoreDrafts(el) {
  const ask = $('#askAny', el); if (ask && drafts.ask) ask.value = drafts.ask;
  const own = $('#ownProblem', el); if (own && drafts.own) own.value = drafts.own;
  $$('form.reply', el).forEach(f => {
    const i = f.querySelector('input');
    const v = drafts.reply[f.dataset.id];
    if (i && v) i.value = v;
  });
  if (ask) ask.addEventListener('input', () => { drafts.ask = ask.value; });
  if (own) own.addEventListener('input', () => { drafts.own = own.value; });
  $$('form.reply', el).forEach(f => {
    const i = f.querySelector('input');
    if (i) i.addEventListener('input', () => { drafts.reply[f.dataset.id] = i.value; });
  });
}
$('#p-debug').addEventListener('submit', e => {
  const own = e.target.closest('#ownProblemForm');
  if (own) {
    e.preventDefault();
    const input = own.querySelector('input');
    const v = input.value.trim();
    if (!v) return;
    input.value = ''; drafts.own = '';
    startDebug('other', v);
    return;
  }
  const f = e.target.closest('form.reply'); if (!f) return;
  e.preventDefault();
  const input = f.querySelector('input');
  const v = input.value.trim(); if (!v) return;
  input.value = ''; delete drafts.reply[f.dataset.id];
  sendFollowup(+f.dataset.id, v);
});
$('#p-debug').addEventListener('click', e => {
  const b = e.target.closest('button[data-act]'); if (!b) return;
  const id = +b.dataset.id;
  if (b.dataset.act === 'resolve') setStatus(id, 'resolved');
  if (b.dataset.act === 'stuck') markStuck(id);
  if (b.dataset.act === 'explain') askHintOrAnswer(id);
  // Undoing an automatic close. The thread goes back to open and the reply
  if (b.dataset.act === 'reopen') {
    const t = state.threads.find(x => x.id === id);
    if (t) { t.autoResolved = false; setStatus(id, 'open'); }
  }
});

let threadSeq = 0;
/* A question about the reading as a whole, with nothing highlighted. Sent as
   problem "other" with the question as the problem text, which is what makes
   the backend aim its reply at exactly what was asked. The whole reading goes
   in as the context paragraph, so the coach is not left treating the question
   as a bare vocabulary word with no story around it. */
function askAboutReading(question) {
  const q = (question || '').trim(); if (!q) return;
  const id = ++threadSeq;
  const whole = reader ? reader.innerText.trim() : '';
  const t = {
    id, problem: 'other', problemText: q,
    text: '', paragraph: whole, status: 'open', busy: true,
    attempts: 0, gaveUp: false, goalHint: goalOverlapWord(q),
    msgs: [{ from: 'ai', text: 'Thinking…', typing: true }], mark: null,
    fromSelection: false
  };
  state.threads.push(t);
  logEvent('debug_start', { id, problem: 'other', problem_text: q, passage: '', from_selection: false });
  dbInsertDebugEvent(t);
  renderDebug();
  askPartner(partnerPayload('debug', t))
    .then(r => { t.busy = false; t.msgs = [{ from: 'ai', text: r.reply, mode: r.mode, demo: r.demo }]; renderDebug(); dbSaveTranscript(t); });
}

function startDebug(problem, problemText) {
  const p = state.pending; if (!p) return;
  const id = ++threadSeq;
  const mark = wrapRange(p.range, id);
  getSelection().removeAllRanges();
  const t = {
    id, problem, problemText: problemText || null,
    text: p.text, paragraph: p.paragraph, status: 'open', busy: true,
    attempts: 0, gaveUp: false, goalHint: goalOverlapWord(p.text),
    msgs: [{ from: 'ai', text: 'Thinking…', typing: true }], mark
  };
  state.threads.push(t);
  state.pending = null;
  logEvent('debug_start', { id, problem, problem_text: problemText || null, passage: p.text });
  dbInsertDebugEvent(t);
  renderDebug();
  askPartner(partnerPayload('debug', t))
    .then(r => { t.busy = false; t.msgs = [{ from: 'ai', text: r.reply, mode: r.mode, demo: r.demo }]; renderDebug(); dbSaveTranscript(t); });
}
function wrapRange(range, id) {
  const mark = document.createElement('mark');
  mark.className = 'dbg'; mark.dataset.id = id; mark.tabIndex = 0;
  try { mark.appendChild(range.extractContents()); range.insertNode(mark); return mark; }
  catch (e) { return null; }
}
async function sendFollowup(id, message) {
  const t = state.threads.find(x => x.id === id); if (!t) return;
  t.attempts = (t.attempts || 0) + 1;
  // The backend decides when to stop asking and explain the key idea
  const mode = 'followup';
  const payload = partnerPayload(mode, t, message);
  t.msgs.push({ from: 'you', text: message });
  t.busy = true; t.msgs.push({ from: 'ai', text: 'Thinking…', typing: true });
  logEvent('debug_reply', { id, message, mode });
  renderDebug();
  const r = await askPartner(payload);
  t.msgs = t.msgs.filter(m => !m.typing); t.msgs.push({ from: 'ai', text: r.reply, mode: r.mode, demo: r.demo }); t.busy = false;
  if (r.mode === 'giveup') t.gaveUp = true;
  // The backend reports mode "success" when the student's reply shows they
  if (r.mode === 'success' && t.status === 'open' && !r.demo) {
    t.autoResolved = true;
    setStatus(id, 'resolved', 'auto');   // re-renders
  } else {
    renderDebug();
  }
  dbSaveTranscript(t);
}
/* `how` records who decided: "student" when they pressed a button, "auto" when the backend classified */
function setStatus(id, status, how = 'student') {
  const t = state.threads.find(x => x.id === id); if (!t) return;
  t.status = status;
  if (t.mark) {
    t.mark.classList.toggle('stuck', status === 'stuck');
    t.mark.classList.toggle('resolved', status === 'resolved');
  }
  const EVENTS = { resolved: 'debug_resolved', stuck: 'debug_stuck', open: 'debug_reopened' };
  logEvent(EVENTS[status] || 'debug_status', { id, how });
  if (status === 'resolved') nudgeSupport(-1);
  else if (status === 'stuck') nudgeSupport(2);
  dbUpdateDebugStatus(t);
  renderDebug();
}
/* True once the coach has actually said something in this thread. The explain
   option only appears after that, so a student sees the first question before
   they can ask to be told. */
function coachHasReplied(t) {
  return (t.msgs || []).some(m => m.from === 'ai' && !m.typing);
}

/* Pressing the explain button offers the Socratic route one more time before
   the answer. Students who reach for the answer often just want it easier, and
   which one they pick is worth recording. */
/* In-app confirm, same shape as the explain dialog. Replaces window.confirm,
   whose box is titled with the browser's own text and looks nothing like the app. */
function askConfirm({ title, body, yes, no, danger }) {
  return new Promise(resolve => {
    let dlg = $('#confirmDlg');
    if (!dlg) {
      dlg = document.createElement('dialog');
      dlg.id = 'confirmDlg';
      document.body.appendChild(dlg);
    }
    dlg.innerHTML = `
      <h3 style="margin:0 0 6px">${esc(title)}</h3>
      <p class="small" style="margin:0 0 14px">${esc(body)}</p>
      <div class="row">
        <button class="btn${danger ? ' danger' : ''}" type="button" id="cfYes">${esc(yes)}</button>
      </div>
      <div class="row" style="margin-top:10px">
        <button class="ghost" type="button" id="cfNo">${esc(no || 'Cancel')}</button>
      </div>`;
    let done = false;
    const finish = (v) => { if (done) return; done = true; try { dlg.close(); } catch (e) {} resolve(v); };
    $('#cfYes', dlg).addEventListener('click', () => finish(true));
    $('#cfNo', dlg).addEventListener('click', () => finish(false));
    dlg.addEventListener('close', () => finish(false), { once: true });
    try { dlg.showModal(); } catch (e) { finish(false); return; }
    // A dialog opens with its first button focused. For something that undoes
    // work, that must not be the button that does it.
    if (danger) $('#cfNo', dlg).focus();
  });
}

function askHintOrAnswer(id) {
  const t = state.threads.find(x => x.id === id); if (!t || t.busy) return;
  let dlg = $('#explainDlg');
  if (!dlg) {
    dlg = document.createElement('dialog');
    dlg.id = 'explainDlg';
    document.body.appendChild(dlg);
  }
  dlg.innerHTML = `
    <h3 style="margin:0 0 6px">Want a hint first, or the answer?</h3>
    <p class="small" style="margin:0 0 14px">A hint points you to the part of the text that helps.
    The answer explains it outright.</p>
    <div class="row">
      <button class="btn" type="button" id="exHint">Give me a hint</button>
      <button class="btn alt" type="button" id="exAnswer">Just explain it</button>
    </div>
    <div class="row" style="margin-top:10px">
      <button class="ghost" type="button" id="exBack">Back to trying</button>
    </div>`;
  const close = () => { try { dlg.close(); } catch (e) {} };
  $('#exHint', dlg).addEventListener('click', () => {
    close(); logEvent('explain_offer', { id, chose: 'hint' }); markStuck(id);
  });
  $('#exAnswer', dlg).addEventListener('click', () => {
    close(); logEvent('explain_offer', { id, chose: 'answer' }); explainThread(id);
  });
  $('#exBack', dlg).addEventListener('click', () => {
    close(); logEvent('explain_offer', { id, chose: 'back' });
  });
  try { dlg.showModal(); } catch (e) { explainThread(id); }
}

/* The student asking to be told outright. Recorded as its own event so the
   rate can be reported, rather than hidden in free text. */
async function explainThread(id) {
  const t = state.threads.find(x => x.id === id); if (!t || t.busy) return;
  t.explained = (t.explained || 0) + 1;
  t.gaveUp = true;
  t.busy = true;
  const says = t.explained === 1
    ? "I don't understand. Can you explain it?"
    : "I still don't understand that explanation.";
  t.msgs.push({ from: 'you', text: says });
  t.msgs.push({ from: 'ai', text: 'Thinking…', typing: true });
  logEvent('debug_explain', { id, times: t.explained });
  renderDebug();
  const r = await askPartner(partnerPayload('explain', t, says));
  t.msgs = t.msgs.filter(m => !m.typing);
  t.msgs.push({ from: 'ai', text: r.reply, mode: r.mode, demo: r.demo });
  t.busy = false;
  renderDebug();
  dbSaveTranscript(t);
}

async function markStuck(id) {
  const t = state.threads.find(x => x.id === id); if (!t) return;
  t.status = 'stuck'; t.busy = true; t.gaveUp = true;
  if (t.mark) t.mark.classList.add('stuck');
  logEvent('debug_stuck', { id });
  dbUpdateDebugStatus(t);
  // Pressing "Still stuck" has to reach the backend as something the
  const stuckSays = "I'm still stuck.";
  t.msgs.push({ from: 'you', text: stuckSays });
  const r = await askPartner(partnerPayload('stuck', t, stuckSays));
  t.msgs.push({ from: 'ai', text: r.reply, mode: r.mode, demo: r.demo }); t.busy = false;
  renderDebug();
  dbSaveTranscript(t);
}
reader.addEventListener('click', e => {
  const m = e.target.closest('mark.dbg'); if (!m) return;
  setTab('debug');
  const card = $(`#p-debug .card[data-id="${m.dataset.id}"]`);
  if (card) { card.scrollIntoView({ block: 'nearest' }); m.classList.add('active'); setTimeout(() => m.classList.remove('active'), 1200); }
});

/* ---- Check-in tab + floating popup ---- */
function onTheReading() {
  const w = $('#readerWorkspace'), v = $('#readingsView');
  return !!(w && !w.hidden && v && !v.hidden);
}
function startCheckins() {
  const c = state.checkin; c.on = true;
  c.lastTick = Date.now(); c.nextAt = c.lastTick + effectiveCheckinSec() * 1000;
}
async function fireCheckin() {
  const c = state.checkin;
  if (c.fetching || c.pending) return;
  c.fetching = true;
  if (currentTab() === 'monitor') renderCheckin();
  const d = await callBackend('/checkin', {
    goal: state.goal && state.goal.text,
    text: reader.innerText.trim().slice(0, 6000),
    struggles: state.threads.map(t => t.text || t.problemText || '').filter(Boolean).slice(-5),
    previous: c.shown.slice(-5),
    response_language: state.responseLanguage === 'auto' ? null : state.responseLanguage
  });
  let prompt, source;
  if (d && d.prompt) { prompt = d.prompt; source = 'ai'; }
  else { prompt = OFFLINE_CHECKINS[c.i++ % OFFLINE_CHECKINS.length]; source = 'demo'; }
  c.fetching = false; c.pending = prompt; c.popupHidden = false; c.shown.push(prompt);
  logEvent('checkin_prompt', { prompt, source });
  updateBadges();
  if (currentTab() === 'monitor') renderCheckin();
  // A popup is more likely to actually get seen than a dot on a tab the
  if (c.popupOn && onTheReading()) showCheckinPopup();
}
function renderCheckin() {
  const c = state.checkin, el = $('#p-monitor');
  el.innerHTML = `
    <h3>Check-ins while you read</h3>
    <p class="lead">A short pause now and then helps you notice if you have drifted.</p>
    <div class="switch">
      <div><b>Remind me</b><div class="small" id="cd"></div></div>
      <button type="button" id="cinToggle" role="switch" aria-checked="${c.on}" aria-label="Turn check-ins on or off"></button>
    </div>
    <div class="switch">
      <div><b>Pop up when ready</b><div class="small">Show it right away instead of waiting for you to open this tab</div></div>
      <button type="button" id="cinPopupToggle" role="switch" aria-checked="${c.popupOn}" aria-label="Turn the check-in popup on or off"></button>
    </div>
    <label class="f" for="cinEvery">How often (baseline)</label>
    <select id="cinEvery">
      ${[30, 60, 90, 180, 300].map(s => `<option value="${s}" ${s === c.everySec ? 'selected' : ''}>Every ${s < 60 ? s + ' seconds' : (s / 60) + (s === 60 ? ' minute' : ' minutes')}</option>`).join('')}
    </select>
    <div class="row" style="margin:12px 0"><button class="btn alt" id="askNow" type="button">Ask me now</button></div>
    <div id="cinCard"></div>
    ${c.responses.length ? `<p class="small">${c.responses.length} check-in${c.responses.length > 1 ? 's' : ''} answered</p>` : ''}
    <p class="small" style="margin-top:16px; padding-top:12px; border-top:1px solid var(--w-line)">
      Check-ins are part of this project: short, occasional pauses that help you notice sooner if you've
      drifted from the text, and help your teacher see where students are getting stuck. Answering honestly
      (including "I'm lost") is what makes them useful -- it's not graded.
    </p>`;

  $('#cinToggle', el).addEventListener('click', () => { c.on ? (c.on = false) : startCheckins(); renderCheckin(); });
  $('#cinPopupToggle', el).addEventListener('click', () => {
    c.popupOn = !c.popupOn;
    try { localStorage.setItem('checkinPopupOn', c.popupOn ? 'on' : 'off'); } catch (e) { /* private browsing etc */ }
    renderCheckin();
  });
  $('#cinEvery', el).addEventListener('change', e => { c.everySec = +e.target.value; if (c.on) c.nextAt = Date.now() + effectiveCheckinSec() * 1000; });
  $('#askNow', el).addEventListener('click', () => { if (c.pending || c.fetching) renderCheckin(); else fireCheckin(); });
  drawCheckinCard();
  updateCountdown();
  updateBadges();
}
/* Shared by the Monitor-tab card and the floating popup */
function checkinCardHtml(prompt) {
  return `
    <b>${esc(prompt)}</b>
    <textarea class="cinAnswer" style="margin-top:8px" placeholder="Type a quick answer (optional)"></textarea>
    <p class="small" style="margin:10px 0 6px">How is it going?</p>
    <div class="row">
      <button class="btn alt" data-level="clear" type="button">I follow it</button>
      <button class="btn alt" data-level="fuzzy" type="button">A bit fuzzy</button>
      <button class="btn alt" data-level="lost" type="button">I'm lost</button>
    </div>`;
}
function wireCheckinCard(box, onAfter) {
  $$('button[data-level]', box).forEach(b => b.addEventListener('click', () => {
    const c = state.checkin;
    const level = b.dataset.level;
    const answerText = ($('.cinAnswer', box) || {}).value?.trim() || '';
    c.responses.push({ level, prompt: c.pending, answer: answerText });
    logEvent('checkin_response', { level, prompt: c.pending, answer: answerText });
    dbInsertCheckin(c.pending, level, answerText);
    nudgeSupport(level === 'clear' ? -1 : level === 'fuzzy' ? 1 : 2);
    c.pending = null;
    if (c.on) c.nextAt = Date.now() + effectiveCheckinSec() * 1000;
    if (currentTab() === 'monitor') renderCheckin(); else updateBadges();
    if (onAfter) onAfter(level);
  }));
}
function drawCheckinCard() {
  const c = state.checkin, box = $('#cinCard'); if (!box) return;
  if (c.fetching) { box.innerHTML = '<p class="small">Thinking of a question…</p>'; return; }
  if (!c.pending) { box.innerHTML = ''; return; }
  box.innerHTML = `<div class="card">${checkinCardHtml(c.pending)}</div>`;
  wireCheckinCard(box, (level) => {
    if (level !== 'clear') {
      $('#cinCard').innerHTML = `<div class="card"><b>Try this</b><p style="margin:6px 0 0">Find the last sentence you understood, then the first one you didn't. Select that spot in the text and debug it.</p>
        <div class="row" style="margin-top:8px"><button class="btn alt" type="button" id="goDebugFromCheckin">Open Debug tab</button></div></div>`;
      const go = $('#goDebugFromCheckin'); if (go) go.addEventListener('click', () => setTab('debug'));
    } else {
      drawCheckinCard();
    }
  });
}
/* Floating, non-blocking popup */
function ensureCheckinPopupHost() {
  let host = $('#checkinPopupHost');
  if (!host) {
    host = document.createElement('div');
    host.id = 'checkinPopupHost';
    document.body.appendChild(host);
  }
  return host;
}
function showCheckinPopup() {
  const c = state.checkin;
  if (!c.pending) return;
  const host = ensureCheckinPopupHost();
  host.innerHTML = `
    <div class="checkin-popup" role="dialog" aria-label="Check-in">
      <button class="checkin-popup-close" type="button" aria-label="Dismiss, answer later">✕</button>
      ${checkinCardHtml(c.pending)}
    </div>`;
  const popup = $('.checkin-popup', host);
  $('.checkin-popup-close', popup).addEventListener('click', () => {
    c.popupHidden = true;   // dismissed, so do not reopen it for this check-in
    host.innerHTML = '';
  });
  wireCheckinCard(popup, () => { host.innerHTML = ''; });
}
function updateCountdown() {
  const c = state.checkin, el = $('#cd'); if (!el) return;
  if (!c.on) { el.textContent = 'Off'; return; }
  if (c.pending) { el.textContent = 'A check-in is waiting'; return; }
  const s = Math.max(0, Math.ceil((c.nextAt - Date.now()) / 1000));
  const clock = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  el.textContent = c.held ? `Next in ${clock}, paused until you are back on the reading`
                          : `Next in ${clock}`;
}
setInterval(() => {
  const c = state.checkin;
  const now = Date.now();
  // Missing means this is the first tick, which is one interval, not zero: with
  // zero a held clock quietly lost a second the first time it was used.
  const since = Math.max(0, now - (c.lastTick || (now - 1000)));
  c.lastTick = now;
  const paint = () => { if (currentTab() === 'monitor') updateCountdown(); };

  // A check-in that arrived while the student was elsewhere waits for them to
  // come back, rather than appearing over whatever they went to do.
  if (c.pending) {
    if (c.popupOn && !c.popupHidden && onTheReading() && !$('.checkin-popup')) showCheckinPopup();
    paint(); return;
  }
  if (!c.on) { paint(); return; }

  // Hold the clock while the student is off the reading, and while a quiz is
  // being written: the backend answers one request at a time, so a check-in
  // started now would sit behind the quiz and come back a minute later.
  c.held = !onTheReading() || quizCtx.status === 'loading';
  if (c.held) { c.nextAt += since; paint(); return; }

  if (now >= c.nextAt && !c.fetching) fireCheckin();
  paint();
}, 1000);

/* ---- Progress tab ---- */
const SUPPORT_LABEL = { steady: 'Steady: check-ins at your normal pace', closer: 'Checking in a bit closer together', extra: 'Extra support: check-ins are more frequent' };
function renderStats() {
  const t = state.threads;
  const resolved = t.filter(x => x.status === 'resolved').length;
  const stuck = t.filter(x => x.status === 'stuck').length;
  const mins = Math.max(1, Math.round((Date.now() - state.startedAt) / 60000));
  const ci = state.checkin.responses;
  const trend = ci.length ? `
    <p class="small" style="margin:14px 0 4px">How check-ins have gone, in order:</p>
    <div class="ci-trend">${ci.map(r => `<i class="ci-dot ${r.level}" title="${r.level}"></i>`).join('')}</div>` : '';
  const g = state.goal;
  const goalRating = !g ? '' : g.finalConfidence == null ? `
    <div class="row" style="margin-top:10px"><button class="btn alt" id="rateGoalBtn" type="button">Rate how well you met your goal</button></div>` : `
    <p class="small" style="margin-top:10px">Expected to understand it: <b>${g.confidence ?? 'N/A'}/5</b> &middot; Actually did: <b>${g.finalConfidence}/5</b></p>`;
  $('#p-stats').innerHTML = `
    <h3>Your session</h3>
    <p class="lead">${state.goal ? 'Goal: ' + esc(state.goal.text) : 'No goal set yet.'}</p>
    <p class="small">Support level: <b>${SUPPORT_LABEL[state.support.level]}</b></p>
    <div class="stats">
      <div class="stat"><b>${t.length}</b><span>Things worked on</span></div>
      <div class="stat"><b>${resolved}</b><span>Cleared up</span></div>
      <div class="stat"><b>${stuck}</b><span>Still stuck</span></div>
      <div class="stat"><b>${ci.length}</b><span>Check-ins answered</span></div>
      <div class="stat stat-wide"><b>${mins} min</b><span>Time in session</span></div>
    </div>
    ${trend}
    ${goalRating}
    <div class="row" style="margin-top:14px">
      <button class="btn" id="endSessionBtn" type="button">End session</button>
      <button class="btn alt" id="startOverBtn" type="button">Start over</button>
      <button class="btn alt" id="exportBtn" type="button">Download session data</button>
    </div>
    <p class="small" style="margin-top:10px">Start over clears your marks and goal but keeps you on this reading. End session takes you back to all readings. The download is a JSON file of everything above.</p>`;
  $('#endSessionBtn').addEventListener('click', endSession);
  $('#exportBtn').addEventListener('click', exportLog);
  $('#startOverBtn').addEventListener('click', async () => {
    const ok = await askConfirm({
      title: 'Start this reading over?',
      body: 'Your marks, goal, and progress for this reading are cleared. The reading itself stays open.',
      yes: 'Start over', no: 'Keep what I have', danger: true
    });
    if (!ok) return;
    resetSession();
    // resetSession wipes the check-in clock, so start it again for the reading still open
    startCheckins();
  });
  const rateBtn = $('#rateGoalBtn');
  if (rateBtn) rateBtn.addEventListener('click', () => {
    $('#p-stats').insertAdjacentHTML('beforeend', `
      <div class="card" id="goalRateCard" style="margin-top:10px">
        <p class="small" style="margin:0 0 8px">How well did you end up understanding it?</p>
        <div class="scale" id="finalConf">${[1,2,3,4,5].map(n => `<button type="button" data-n="${n}">${n}</button>`).join('')}</div>
      </div>`);
    $$('#finalConf button').forEach(b => b.addEventListener('click', () => {
      g.finalConfidence = +b.dataset.n;
      logEvent('goal_final_rating', { confidence: g.confidence, finalConfidence: g.finalConfidence });
      if (state.session.id) dbUpdateSession({ final_confidence: g.finalConfidence });
      renderStats();
    }));
  });
}
function exportLog() {
  const data = { exportedAt: new Date().toISOString(), goal: state.goal, events: state.log };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'reading-session.json';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 500);
}
function resetSession() {
  $$('mark.dbg', reader).forEach(m => { const p = m.parentNode; while (m.firstChild) p.insertBefore(m.firstChild, m); p.removeChild(m); p.normalize(); });
  Object.assign(state, {
    goal: null, threads: [], pending: null, selectionNote: '',
    session: { id: null },   // a new reading means a new database session
    pastedTitle: null,
    startedAt: Date.now(), log: []
  });
  Object.assign(state.checkin, { on: false, pending: null, responses: [], i: 0, fetching: false, shown: [] });
  Object.assign(state.support, { score: 0, level: 'steady' });
  clearDrafts();
  $('#goalLine').textContent = 'No goal set yet';
  renderGoal(); renderDebug(); setTab('goal'); updateBadges();
  const host = $('#checkinPopupHost'); if (host) host.innerHTML = '';
}
/* "End session" leaves this reading entirely and returns to the picker */
async function endSession() {
  const ok = await askConfirm({
    title: 'End this session?',
    body: 'This closes the reading and takes you back to all readings. Download your session data first if you want to keep it.',
    yes: 'End session', no: 'Keep reading', danger: true
  });
  if (!ok) return;
  resetSession();
  showReadingPicker();
}

function setReaderText(title, byline, bodyParagraphsHtml) {
  $('#docTitle').textContent = title;
  $('#docBy').textContent = byline;
  reader.innerHTML = bodyParagraphsHtml;
}
function paragraphsToHtml(raw) {
  return raw.split(/\n\s*\n/).map(p => `<p>${esc(p.replace(/\s*\n\s*/g, ' ').trim())}</p>`).join('');
}

/* ---------- */
let pasteTarget = 'reader';
function openPasteDialog(target) {
  pasteTarget = target;
  $('#pasteFileMsg').textContent = '';
  $('#pasteDlg').showModal();
}

/* The name the student gave their own text */
function pastedTitleOrDefault() {
  const typed = ($('#pasteTitle') ? $('#pasteTitle').value : '').trim();
  return typed.slice(0, 120) || 'Your text';
}
$('#pasteBtn').addEventListener('click', () => openPasteDialog('reader'));

$('#pasteFile').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = ''; // so picking the same file twice still fires "change"
  if (!file) return;
  const msg = $('#pasteFileMsg');
  msg.style.color = 'var(--muted)';
  msg.textContent = `Reading ${file.name}…`;
  try {
    const text = await extractTextFromFile(file);
    $('#pasteText').value = text;
    // Only when they have not named it themselves, so an upload never
    const titleBox = $('#pasteTitle');
    if (titleBox && !titleBox.value.trim()) {
      titleBox.value = file.name.replace(/\.[^.]+$/, '').slice(0, 120);
    }
    msg.style.color = 'var(--ok)';
    msg.textContent = `Loaded text from ${file.name}. Review it below, then click "Load text".`;
  } catch (err) {
    msg.style.color = 'var(--bug)';
    msg.textContent = err.message || 'Could not read that file.';
  }
});
/* Listen on the form's submit rather than the button's click */
$('#pasteCancel').addEventListener('click', () => $('#pasteDlg').close('cancel'));
$('#pasteDlg').querySelector('form').addEventListener('submit', (e) => {
  e.preventDefault();
  const raw = $('#pasteText').value.trim();
  if (!raw) {
    // Nothing to load. Say so rather than closing on an empty dialog.
    $('#pasteFileMsg').style.color = 'var(--bug)';
    $('#pasteFileMsg').textContent = 'Paste some text or upload a file first.';
    return;
  }
  const title = pastedTitleOrDefault();
  $('#pasteText').value = '';
  if ($('#pasteTitle')) $('#pasteTitle').value = '';
  $('#pasteDlg').close('ok');
  if (pasteTarget === 'reader') {
    enterReader({ title, byline: 'Your own text', bodyHtml: paragraphsToHtml(raw), readingId: null });
  } else {
    setQuizCtx({ readingId: null, isDemo: false, demoId: null, isPasted: true, title, text: raw, sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null, quizId: null, instructions: null });
    renderQuizWorkspace();
  }
});

/* Sidebar navigation: Readings <-> Quizzes <-> Settings Each nav button's data-view matches a */
function showView(name) {
  if (!document.getElementById(name + 'View')) {
    showToast('That screen is missing from this page. The index.html on the server is older than app.js.', 'bug');
    return;
  }
  $$('.side-link').forEach(b => {
    const active = b.dataset.view === name;
    b.classList.toggle('active', active);
    b.setAttribute('aria-current', active ? 'page' : 'false');
  });
  $$('.view').forEach(v => { v.hidden = v.id !== name + 'View'; });
  // Re-read the reading list on the way in, so a class joined a moment ago
  // shows up without a page reload. An open reading is left alone.
  if (name === 'readings' && !$('#pickerPane').hidden) showReadingPicker();
  if (name === 'quizzes') renderQuizzesView();
  if (name === 'classes') renderClassesView();
  if (name === 'history') renderHistoryView();
  if (name === 'settings') renderSettings('settingsPane', state.user);
}
$$('.side-link').forEach(b => b.addEventListener('click', () => showView(b.dataset.view)));

/* CLASSES: join a teacher's class with a code */
let openClassId = null;
/* A render that reads the database can be overtaken by a newer one. Each pane
   keeps a counter; a render that is no longer the latest stops before it paints,
   so a slow older render cannot put stale cards back on the screen. */
const renderTokens = { classes: 0, readings: 0, quizzes: 0 };
function newRender(pane) { return ++renderTokens[pane]; }
function stillCurrent(token, pane = 'classes') { return token === undefined || token === renderTokens[pane]; }

async function renderClassesView() {
  const el = $('#classesPane');
  if (!el) {
    showToast('The Classes screen is missing from this page. The index.html on the server is older than app.js.', 'bug');
    return;
  }
  el.innerHTML = `<h1>Your classes</h1><p class="hint">Loading…</p>`;
  // Two renders can overlap (click a class, then Back before it finishes). The
  // later one wins, so a finished older one does not paint over it.
  const token = newRender('classes');
  try {
    if (openClassId) await renderClassDetail(el, openClassId, token);
    else await renderClassesViewInner(el, token);
  } catch (err) {
    if (!stillCurrent(token)) return;
    el.innerHTML = `<h1>Your classes</h1><p class="small" style="color:var(--bug)">Could not load your classes: ${esc((err && err.message) || String(err))}</p>`;
  }
}

async function renderClassDetail(el, classId, token) {
  const backBtn = '<button class="ghost" id="clsBack" type="button">&larr; All classes</button>';
  const wireBack = () => {
    const b = $('#clsBack', el);
    if (b) b.addEventListener('click', () => { openClassId = null; renderClassesView(); });
  };

  const { data: cls, error: clsErr } = await supa
    .from('classes').select('id, name, teacher_id').eq('id', classId).maybeSingle();

  if (!stillCurrent(token)) return;
  if (clsErr || !cls) {
    el.innerHTML = `<div class="row">${backBtn}</div>
      <h1>Class</h1>
      <p class="small" style="color:var(--bug)">Could not open that class: ${esc(clsErr ? clsErr.message : 'it may have been deleted.')}</p>`;
    wireBack();
    return;
  }

  let teacherName = 'Teacher';
  if (cls.teacher_id) {
    const { data: t } = await supa.from('profiles').select('full_name').eq('id', cls.teacher_id).maybeSingle();
    if (t && t.full_name) teacherName = t.full_name;
  }

  const SEL = 'id, title, body, created_at';
  let forClass = [], general = [], degraded = false, loadErr = null;

  let res = await supa.from('readings').select(SEL)
    .eq('class_id', classId).order('created_at', { ascending: false });

  if (res.error && isMissingColumn(res.error, 'class_id')) {
    degraded = true;
    res = await supa.from('readings').select(SEL)
      .eq('teacher_id', cls.teacher_id).order('created_at', { ascending: false });
  }
  if (res.error) loadErr = res.error; else forClass = res.data || [];

  if (!degraded && !loadErr && cls.teacher_id) {
    const g = await supa.from('readings').select(SEL)
      .eq('teacher_id', cls.teacher_id).is('class_id', null)
      .order('created_at', { ascending: false });
    if (!g.error) general = g.data || [];
  }

  let main;
  if (loadErr) {
    main = `<p class="small" style="color:var(--bug)">Could not load this class's readings: ${esc(loadErr.message)}</p>`;
  } else if (forClass.length) {
    main = readingCardsHtml(forClass, BOOK_ICON);
  } else {
    main = `<p class="small">No readings have been added to this class yet.</p>`;
  }

  if (!stillCurrent(token)) return;
  el.innerHTML = `
    <div class="row">${backBtn}</div>
    <h1 style="margin-top:10px">${esc(cls.name)}</h1>
    <p class="hint">Taught by ${esc(teacherName)}</p>
    ${degraded ? `<p class="small" style="color:var(--bug)">Readings cannot be split by class yet, so this shows everything from this teacher. Run readings_class_id_migration.sql to fix that.</p>` : ''}
    <h3 style="margin:22px 0 0">${degraded ? 'Readings from this teacher' : 'Readings for this class'}</h3>
    ${main}
    ${general.length ? `<h3 style="margin:28px 0 0">Also from this teacher</h3>
    <p class="small" style="margin:4px 0 0">Not tied to one class, so every student of this teacher can read them.</p>
    ${readingCardsHtml(general, BOOK_ICON)}` : ''}`;

  wireBack();

  $$('.lib-card[data-id]', el).forEach(b => b.addEventListener('click', async () => {
    const r = await fetchReadingForCard(b);
    if (!r) return;
    showView('readings');
    enterReader({
      title: r.title,
      byline: 'Added by your teacher',
      bodyHtml: paragraphsToHtml(r.body),
      readingId: r.id,
      // So "Back to <class>" returns to the class page the student came from
      from: { kind: 'class', id: classId, name: cls.name }
    });
  }));
}

async function renderClassesViewInner(el, token) {
  const { data: memberships, error } = await supa
    .from('class_members')
    .select('class_id')          // joined_at was selected and never used, and breaks if the column is absent
    .eq('student_id', state.user.id);

  if (!stillCurrent(token)) return;
  if (error) {
    el.innerHTML = `<h1>Your classes</h1><p class="small" style="color:var(--bug)">Could not read your class list: ${esc(error.message)}</p>`;
    return;
  }

  const classIds = [...new Set((memberships || []).map(m => m.class_id).filter(Boolean))];

  let classRows = [];
  if (classIds.length) {
    const { data: cls, error: clsErr } = await supa
      .from('classes').select('id, name, teacher_id').in('id', classIds);
    if (clsErr) {
      el.innerHTML = `<h1>Your classes</h1><p class="small" style="color:var(--bug)">Could not read the classes you joined: ${esc(clsErr.message)}</p>`;
      return;
    }
    classRows = cls || [];
  }

  const teacherIds = [...new Set(classRows.map(c => c.teacher_id).filter(Boolean))];
  let teacherNames = {};
  if (teacherIds.length) {
    const { data: teachers } = await supa.from('profiles').select('id, full_name').in('id', teacherIds);
    (teachers || []).forEach(t => { teacherNames[t.id] = t.full_name || 'Teacher'; });
  }

  const rows = classRows.map(c => `
    <div class="reading-row-card" data-class="${c.id}" role="button" tabindex="0" style="cursor:pointer">
      <div>
        <b>${esc(c.name)}</b>
        <div class="small">Taught by ${esc(teacherNames[c.teacher_id] || 'Teacher')}</div>
      </div>
      <span class="small" aria-hidden="true">View readings &rarr;</span>
      <button class="tiny danger" type="button" data-act="leave">Leave</button>
    </div>`).join('');

  const orphaned = classIds.length - classRows.length;

  if (!stillCurrent(token)) return;
  el.innerHTML = `
    <h1>Your classes</h1>
    <p class="hint">Joining a class lets you see the readings that teacher adds. Ask your teacher for their class's join code.</p>

    <div class="card" style="max-width:420px">
      <h3 style="margin:0 0 10px">Join a class</h3>
      <label class="f" for="joinCodeInput">Join code</label>
      <input type="text" id="joinCodeInput" placeholder="e.g. 7K4PXM" maxlength="8" style="text-transform:uppercase; letter-spacing:2px">
      <div class="row" style="margin-top:12px">
        <button class="btn" id="joinClassBtn" type="button">Join class</button>
      </div>
    </div>

    ${rows ? `<div style="margin-top:18px; display:flex; flex-direction:column; gap:10px">${rows}</div>`
           : '<p class="small" style="margin-top:16px">You haven\'t joined any classes yet. The demo story and pasting your own text still work either way.</p>'}
    ${orphaned > 0 ? `<p class="small" style="margin-top:12px; color:var(--bug)">${orphaned} class${orphaned === 1 ? '' : 'es'} you joined could not be read back. Tell your teacher the class may have been deleted.</p>` : ''}`;

  $('#joinClassBtn').addEventListener('click', async () => {
    const codeInput = $('#joinCodeInput');
    const code = codeInput.value.trim().toUpperCase();
    if (!code) return;
    const { data: cls, error: findErr } = await supa.from('classes').select('id, name').eq('join_code', code).maybeSingle();
    if (findErr || !cls) { showToast("That join code didn't match a class. Double-check it with your teacher.", 'bug'); return; }
    const { error: joinErr } = await supa.from('class_members').insert({ class_id: cls.id, student_id: state.user.id });
    if (joinErr) {
      showToast(/duplicate|unique/i.test(joinErr.message) ? "You're already in that class." : 'Could not join: ' + joinErr.message, 'bug');
      if (/duplicate|unique/i.test(joinErr.message)) { codeInput.value = ''; openClassId = cls.id; renderClassesView(); }
      return;
    }
    codeInput.value = '';
    showToast('Joined ' + (cls.name || 'the class') + '.', 'ok');
    // Land on the class page, so the readings that come with the class are
    // visible straight away instead of needing a reload.
    openClassId = cls.id;
    renderClassesView();
  });

  const openClass = (id) => { openClassId = id; renderClassesView(); };

  el.querySelectorAll('[data-class][role="button"]').forEach(card => {
    card.addEventListener('click', (e) => {
      if (e.target.closest('[data-act="leave"]')) return;
      openClass(card.dataset.class);
    });
    card.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      e.preventDefault();
      openClass(card.dataset.class);
    });
  });

  el.querySelectorAll('[data-act="leave"]').forEach(btn => btn.addEventListener('click', async (e) => {
    e.stopPropagation();
    const row = btn.closest('[data-class]');
    const name = (row.querySelector('b') || {}).textContent || 'this class';
    const ok = await askConfirm({
      title: 'Leave ' + name + '?',
      body: "You will not see that teacher's readings until you rejoin with the class code.",
      yes: 'Leave class', no: 'Stay in the class', danger: true
    });
    if (!ok) return;
    const { error: leaveErr } = await supa.from('class_members').delete().eq('class_id', row.dataset.class).eq('student_id', state.user.id);
    if (leaveErr) { showToast('Could not leave: ' + leaveErr.message, 'bug'); return; }
    showToast('Left the class.', 'ok');
    renderClassesView();
  }));
}

/* HISTORY: every past session for this student, own data only */
async function renderHistoryView() {
  const el = $('#historyPane');
  el.innerHTML = `<h1>Your history</h1><p class="hint">Loading…</p>`;

  const FIELDS = 'id, goal, confidence, final_confidence, created_at, readings(title)';
  let { data: sessions, error } = await supa
    .from('sessions')
    .select(FIELDS + ', title')
    .eq('student_id', state.user.id)
    .order('created_at', { ascending: false });

  // Without session_title_migration.sql there is no title column, and asking
  if (error && isMissingColumn(error, 'title')) {
    ({ data: sessions, error } = await supa
      .from('sessions')
      .select(FIELDS)
      .eq('student_id', state.user.id)
      .order('created_at', { ascending: false }));
  }

  if (error) {
    // A missing column is a setup problem, not something the student did
    const needsMigration = /created_at|column .* does not exist/i.test(error.message);
    el.innerHTML = `<h1>Your history</h1><p class="small" style="color:var(--bug)">Could not load your history: ${esc(error.message)}${
      needsMigration ? '. Run sessions_created_at_migration.sql in Supabase to add the missing column.' : ''
    }</p>`;
    return;
  }
  if (!sessions || !sessions.length) {
    el.innerHTML = `<h1>Your history</h1><p class="hint">Once you start a reading, your past sessions will show up here.</p>`;
    return;
  }

  // Two batched queries instead of one per session row.
  const ids = sessions.map(s => s.id);
  const [{ data: debugRows }, { data: checkinRows }] = await Promise.all([
    supa.from('debug_events').select('session_id, status').in('session_id', ids),
    supa.from('checkin_events').select('session_id, level').in('session_id', ids)
  ]);
  const statsFor = (id) => {
    const d = (debugRows || []).filter(r => r.session_id === id);
    const c = (checkinRows || []).filter(r => r.session_id === id);
    return { resolved: d.filter(r => r.status === 'resolved').length, stuck: d.filter(r => r.status === 'stuck').length, total: d.length, checkins: c.length };
  };

  const rows = sessions.map(s => {
    const st = statsFor(s.id);
    const when = s.created_at ? new Date(s.created_at).toLocaleDateString() : '';
    // A session points at a reading, or carries its own title for text the
    const title = (s.readings && s.readings.title) || s.title || 'A pasted or deleted text';
    return `<div class="reading-row-card">
      <div>
        <b>${esc(title)}</b>
        <div class="small">
          ${when ? when + ' &middot; ' : ''}${s.goal ? 'Goal: ' + esc(s.goal) + ' &middot; ' : ''}
          ${st.total} spot${st.total === 1 ? '' : 's'} debugged (${st.resolved} cleared up, ${st.stuck} still stuck)
          &middot; ${st.checkins} check-in${st.checkins === 1 ? '' : 's'}
          ${s.confidence != null && s.final_confidence != null ? ` &middot; Confidence ${s.confidence} &rarr; ${s.final_confidence}` : ''}
        </div>
      </div>
    </div>`;
  }).join('');

  el.innerHTML = `
    <h1>Your history</h1>
    <p class="hint">Every reading session you've started, most recent first.</p>
    <div style="display:flex; flex-direction:column; gap:10px; margin-top:12px">${rows}</div>`;
}

function handleLogin(profile) {
  state.user = profile; // { id, role, full_name }
  $('#userName').textContent = profile.full_name || 'Student';
  showView('readings');
  showReadingPicker();
}

/* READINGS: library grid -> reader + side panel */
function readingCardsHtml(readings, iconSvg, opts = {}) {
  return `<div class="lib-grid">${readings.map(r => {
    const words = r.body.trim().split(/\s+/).length;
    const mins = Math.max(1, Math.round(words / 150));
    const snippet = r.body.replace(/\s+/g, ' ').trim().slice(0, 110);
    const cls = opts.demo ? 'lib-card demo' : 'lib-card';
    const attr = opts.demo ? `data-demo="${r.id}"` : `data-id="${r.id}"`;
    return `<button class="${cls}" ${attr}>
      ${opts.demo || opts.builtIn ? '<span class="lib-card-badge">Built-in</span>' : ''}
      <span class="lib-card-icon">${iconSvg}</span>
      <span class="lib-card-title">${esc(r.title)}</span>
      <span class="lib-card-snippet">${esc(snippet)}…</span>
      <span class="lib-card-meta">${mins} min read</span>
    </button>`;
  }).join('')}</div>`;
}
/* Fetches one reading row for a card click. Returns null and says so on the
   card itself if the row cannot be read, instead of leaving a dead button. */
async function fetchReadingForCard(btn) {
  btn.disabled = true;
  const old = btn.querySelector('.lib-card-fail');
  if (old) old.remove();
  const { data, error } = await supa.from('readings').select('*').eq('id', btn.dataset.id).single();
  if (error || !data) {
    btn.disabled = false;
    btn.insertAdjacentHTML('beforeend',
      `<span class="lib-card-fail">Could not open this reading. Try again, or ask your teacher whether it is still shared.</span>`);
    btn.scrollIntoView({ block: 'nearest' });
    console.warn('Could not open reading', btn.dataset.id, error && error.message);
    return null;
  }
  return data;
}
/* Built-in demo readings reuse the same card markup */
function demoReadingObjs() { return DEMO_READINGS.map(d => ({ id: d.id, title: d.title, body: d.text })); }
const BOOK_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 5.2c2.3-1 5-1 7 .3v12.8c-2-1.3-4.7-1.3-7-.3V5.2Z"/><path d="M18.5 5.2c-2.3-1-5-1-7 .3v12.8c2-1.3 4.7-1.3 7-.3V5.2Z"/></svg>`;

/* Readings a student can actually see: general readings */
async function getVisibleReadings() {
  if (!state.user) return { data: [], error: null, joinedCount: 0 };

  const { data: memberships, error: memErr } = await supa
    .from('class_members').select('class_id').eq('student_id', state.user.id);
  if (memErr) return { data: null, error: memErr, stage: 'membership', joinedCount: 0 };

  const classIds = (memberships || []).map(m => m.class_id).filter(Boolean);

  const base = (cols) => supa.from('readings').select(cols)
    .not('teacher_id', 'is', null)
    .order('created_at', { ascending: false });
  const FULL = 'id, title, body, created_at, class_id';

  let res;
  if (classIds.length) {
    const list = classIds.map(id => '"' + id + '"').join(',');
    res = await base(FULL).or('class_id.is.null,class_id.in.(' + list + ')');
  } else {
    res = await base(FULL).is('class_id', null);
  }

  if (res.error && isMissingColumn(res.error, 'class_id')) {
    res = await base('id, title, body, created_at');
    if (!res.error) res.degraded = 'class_id';
  }

  res.joinedCount = classIds.length;
  res.stage = 'readings';
  return res;
}

/* Builds the two teacher sections. A reading saved to a class the student joined
   goes under "From your classes"; one with no class goes under "From your
   teachers", because filing those under a class was misleading. */
function teacherReadingsHtml(res, icon, linkId, opts = {}) {
  if (res.error || !res.data || !res.data.length) {
    return `<h3 style="margin:28px 0 0">From your classes</h3>${readingsEmptyHtml(res, linkId)}`;
  }
  if (res.degraded === 'class_id') {
    return `<h3 style="margin:28px 0 0">From your teachers</h3>
      <p class="small" style="margin:4px 0 0">Readings cannot be split by class on this database yet.</p>
      ${readingCardsHtml(res.data, icon, opts)}`;
  }
  const inClass = res.data.filter(r => r.class_id);
  const general = res.data.filter(r => !r.class_id);
  let out = '';
  if (inClass.length) {
    out += `<h3 style="margin:28px 0 0">From your classes</h3>${readingCardsHtml(inClass, icon, opts)}`;
  }
  if (general.length) {
    out += `<h3 style="margin:28px 0 0">From your teachers</h3>
      <p class="small" style="margin:4px 0 0">Not tied to one class, so every student of that teacher can read them.</p>
      ${readingCardsHtml(general, icon, opts)}`;
  }
  return out;
}

function readingsEmptyHtml(res, linkId) {
  if (res.error) {
    const where = res.stage === 'membership' ? 'your class list' : 'the readings';
    return '<p class="small" style="color:var(--bug)">Could not load ' + where + ': '
      + esc(res.error.message || 'unknown error') + '</p>';
  }
  if (!res.joinedCount) {
    return '<p class="small">You have not joined a class yet. <a href="#" id="' + linkId
      + '">Join a class</a> with the code from your teacher to see the readings they add.</p>';
  }
  return '<p class="small">Your class has no readings yet. Either your teacher has not added one, '
    + 'or they saved it to a different class.</p>';
}

/* The built-in practice readings, from the database rather than the copies below */
async function getBuiltInReadings() {
  const { data, error } = await supa
    .from('readings').select('id, title, body')
    .is('teacher_id', null).order('created_at');
  if (error || !data || !data.length) {
    if (error) console.warn('Falling back to the built-in copies:', error.message);
    return { rows: demoReadingObjs(), fromDb: false };
  }
  return { rows: data, fromDb: true };
}

/* Where the open reading was picked, so the reader's back button goes back there */
let readerFrom = { kind: 'readings' };
$('#changeReadingBtn').addEventListener('click', () => {
  if (readerFrom.kind === 'class') {
    $('#readerWorkspace').hidden = true;
    $('#pickerPane').hidden = false;
    openClassId = readerFrom.id;
    showView('classes');
    return;
  }
  showReadingPicker();
});

async function showReadingPicker() {
  $('#readerWorkspace').hidden = true;
  const picker = $('#pickerPane');
  picker.hidden = false;
  picker.innerHTML = `<h1>Choose a reading</h1><p class="hint">Loading…</p>`;
  const token = newRender('readings');

  const res = await getVisibleReadings();
  const builtIn = await getBuiltInReadings();
  if (!stillCurrent(token, 'readings')) return;

  picker.innerHTML = `
    <h1>Choose a reading</h1>
    <p class="hint">Pick a built-in practice text, something your teacher added, or paste your own.</p>
    <h3 style="margin:22px 0 0">Built-in practice readings</h3>
    ${readingCardsHtml(builtIn.rows, DEMO_READING_ICON, { demo: !builtIn.fromDb, builtIn: true })}
    ${teacherReadingsHtml(res, BOOK_ICON, 'goToClasses')}
    <div class="row" style="margin-top:16px">
      <button class="btn alt" id="pickPaste" type="button">Paste or upload my own text</button>
    </div>`;

  $$('.lib-card[data-id]', picker).forEach(b => b.addEventListener('click', async () => {
    const r = await fetchReadingForCard(b);
    if (!r) return;
    // Both grids on this page render data-id cards, so the byline has to come
    enterReader({
      title: r.title,
      byline: r.teacher_id ? 'Added by your teacher' : 'Built-in practice text',
      bodyHtml: paragraphsToHtml(r.body),
      readingId: r.id
    });
  }));
  $$('.lib-card[data-demo]', picker).forEach(b => b.addEventListener('click', () => {
    const d = DEMO_READINGS.find(x => x.id === b.dataset.demo);
    if (!d) return;
    enterReader({ title: d.title, byline: 'Built-in practice text', bodyHtml: paragraphsToHtml(d.text), readingId: null });
  }));
  $('#pickPaste', picker).addEventListener('click', () => openPasteDialog('reader'));
  const classesLink = $('#goToClasses', picker);
  if (classesLink) classesLink.addEventListener('click', (e) => { e.preventDefault(); showView('classes'); });
}

function enterReader({ title, byline, bodyHtml, readingId, from }) {
  resetSession();
  setReaderText(title, byline, bodyHtml);
  state.reading = { id: readingId };
  // Where the reader was opened from, so the back button returns there
  readerFrom = from || { kind: 'readings' };
  const back = $('#changeReadingBtn');
  if (back) back.innerHTML = readerFrom.kind === 'class'
    ? '&larr; Back to ' + esc(readerFrom.name || 'class')
    : '&larr; All readings';
  // Set after resetSession, which clears it. Only a text with no reading row
  state.pastedTitle = readingId ? null : title;

  $('#pickerPane').hidden = true;
  $('#readerWorkspace').hidden = false;
  $('#pasteBtn').hidden = false;

  renderGoal(); renderDebug();
  // Start the clock with the reading, not with the goal, so skipping the goal
  // popup does not leave check-ins switched off for the whole session.
  startCheckins();
  showGoalPopup();
}

/* QUIZZES: its own top-level view, decoupled from the reading workspace */
/* Bumped whenever the quiz screen changes, so a slow question request that lands
   after the student backed out or picked another reading is dropped, not painted. */
let quizRequestToken = 0;
/* Every switch of what the quiz screen is about goes through here */
function setQuizCtx(next) { quizRequestToken++; clearInterval(quizWaitTimer); quizCtx = next; }

let quizCtx = { readingId: null, isDemo: false, demoId: null, isPasted: false, title: '', text: '', sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null, quizId: null, instructions: null };

async function ensureQuizSession() {
  if (quizCtx.sessionId) return quizCtx.sessionId;
  if (!state.user) return null;
  // kind: 'quiz' keeps this out of the reading-session counts. A quiz
  const row = { student_id: state.user.id, reading_id: quizCtx.readingId, kind: 'quiz' };
  if (!quizCtx.readingId && quizCtx.isPasted && quizCtx.title) row.title = quizCtx.title;

  let { data, error } = await supa.from('sessions').insert(row).select('id').single();
  if (error && row.title && isMissingColumn(error, 'title')) {
    delete row.title;
    ({ data, error } = await supa.from('sessions').insert(row).select('id').single());
  }
  if (error) { noteSaveFailure('the quiz attempt', error); return null; }
  quizCtx.sessionId = data.id;
  return data.id;
}
async function saveQuizAnswer(question, chosen, correct, questionId) {
  const sid = await ensureQuizSession(); if (!sid) return;
  const { error } = await supa.from('quiz_results').insert({
    session_id: sid, question, chosen, correct,
    quiz_id: quizCtx.quizId, question_id: questionId || null });
  if (error) noteSaveFailure('this quiz answer', error);
}

function renderQuizzesView() {
  if (quizCtx.quizId || quizCtx.readingId || quizCtx.isDemo || quizCtx.isPasted) renderQuizWorkspace();
  else renderQuizPicker();
}

const QUIZ_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="5.5" y="4.5" width="13" height="16" rx="2"/><path d="M9 4.5V3.3a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1V4.5M8.5 12l2 2 4.5-4.5"/></svg>`;

async function renderQuizPicker() {
  const el = $('#quizzesPane');
  el.innerHTML = `<h1>Quizzes</h1><p class="hint">Loading…</p>`;
  const token = newRender('quizzes');

  const res = await getVisibleReadings();

  const assigned = await getAssignedQuizzes();
  const builtIn = await getBuiltInReadings();
  if (!stillCurrent(token, 'quizzes')) return;

  el.innerHTML = `
    <h1>Quizzes</h1>
    <p class="hint">Pick a reading to test yourself on: a built-in text, something your teacher added, or your own paste.</p>
    ${assigned.length ? `<h3 style="margin:22px 0 0">Assigned by your teacher</h3>
    <div class="quiz-list">${assigned.map(z => `
      <button class="quiz-row" data-assigned="${z.id}">
        <span class="quiz-row-icon">${QUIZ_ICON}</span>
        <span class="quiz-row-body">
          <span class="quiz-row-title">${esc(z.title)}<span class="quiz-row-badge">Assigned</span></span>
          <span class="quiz-row-meta">${z.n} question${z.n === 1 ? '' : 's'}${z.instructions ? ' &middot; ' + esc(z.instructions) : ''}</span>
        </span>
        <span class="quiz-row-cta">Start quiz &rarr;</span>
      </button>`).join('')}</div>` : ''}
    <h3 style="margin:22px 0 0">Built-in practice texts</h3>
    ${readingCardsHtml(builtIn.rows, DEMO_READING_ICON, { demo: !builtIn.fromDb, builtIn: true, kind: 'quiz' })}
    ${teacherReadingsHtml(res, QUIZ_ICON, 'goToClassesQuiz')}
    <div class="row" style="margin-top:16px">
      <button class="btn alt" id="quizPaste" type="button">Paste or upload my own text</button>
    </div>`;

  $$('.lib-card[data-id]', el).forEach(b => b.addEventListener('click', async () => {
    const r = await fetchReadingForCard(b);
    if (!r) return;
    setQuizCtx({ readingId: r.id, isDemo: false, demoId: null, isPasted: false, title: r.title, text: r.body, sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null, quizId: null, instructions: null });
    renderQuizWorkspace();
  }));
  $$('.lib-card[data-demo]', el).forEach(b => b.addEventListener('click', () => {
    const d = DEMO_READINGS.find(x => x.id === b.dataset.demo);
    if (!d) return;
    setQuizCtx({ readingId: null, isDemo: true, demoId: d.id, isPasted: false, title: d.title, text: d.text, sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null, quizId: null, instructions: null });
    renderQuizWorkspace();
  }));
  $$('button[data-assigned]', el).forEach(b => b.addEventListener('click', async () => {
    b.disabled = true;
    await openAssignedQuiz(b.dataset.assigned);
  }));
  $('#quizPaste', el).addEventListener('click', () => openPasteDialog('quiz'));
  const classesLink = $('#goToClassesQuiz', el);
  if (classesLink) classesLink.addEventListener('click', (e) => { e.preventDefault(); showView('classes'); });
}

/* Quizzes a teacher has assigned to a class this student has joined */
async function getAssignedQuizzes() {
  if (!state.user) return [];
  const { data, error } = await supa
    .from('quizzes')
    .select('id, title, instructions, created_at, quiz_questions(id)')
    .order('created_at', { ascending: false });
  if (error) { console.warn('Could not load assigned quizzes:', error.message); return []; }
  return (data || [])
    .map(z => ({ id: z.id, title: z.title, instructions: z.instructions,
                 n: (z.quiz_questions || []).length }))
    .filter(z => z.n > 0);
}

/* Answers this student already gave on the assigned quiz now open */
async function loadPreviousAnswers() {
  if (!quizCtx.quizId || !state.user) return;
  const { data, error } = await supa
    .from('quiz_results')
    .select('question_id, chosen, created_at, sessions!inner(student_id)')
    .eq('quiz_id', quizCtx.quizId)
    .eq('sessions.student_id', state.user.id)
    .order('created_at', { ascending: true });
  if (error) { console.warn('Could not load earlier answers:', error.message); return; }

  const firstFor = {};
  (data || []).forEach(row => {
    if (row.question_id && !(row.question_id in firstFor)) firstFor[row.question_id] = row.chosen;
  });
  quizCtx.questions.forEach((q, i) => {
    if (!(q.id in firstFor)) return;
    const j = q.options.findIndex(o => o === firstFor[q.id]);
    if (j > -1) quizCtx.answers[i] = j;
  });
}

/* An assigned quiz is answered exactly as the teacher assigned it, so its questions are loaded rather */
async function openAssignedQuiz(quizId) {
  const el = $('#quizzesPane');
  el.innerHTML = `<h1>Quizzes</h1><p class="hint">Opening the quiz…</p>`;
  const [{ data: quiz, error: zErr }, { data: rows, error: qErr }] = await Promise.all([
    supa.from('quizzes').select('id, title, instructions, reading_id').eq('id', quizId).single(),
    supa.from('quiz_questions').select('id, position, kind, q, options, answer, why, evidence')
        .eq('quiz_id', quizId).order('position')
  ]);
  if (zErr || qErr || !quiz || !rows || !rows.length) {
    const e = zErr || qErr;
    el.innerHTML = `<h1>Quizzes</h1><p class="small" style="color:var(--bug)">Could not open that quiz${
      e ? ': ' + esc(e.message) : '.'}</p>`;
    return;
  }
  setQuizCtx({
    readingId: quiz.reading_id, isDemo: false, demoId: null, isPasted: false,
    title: quiz.title, text: '', sessionId: null,
    questions: rows.map(r => ({ id: r.id, kind: r.kind, q: r.q, options: r.options,
                                answer: r.answer, why: r.why, evidence: r.evidence })),
    answers: {}, status: 'ready', error: '', source: 'teacher',
    quizId: quiz.id, instructions: quiz.instructions
  });
  // Show what they answered before rather than a blank quiz. Only the first
  await loadPreviousAnswers();
  renderQuizWorkspace();
}

/* How many questions a passage can carry */
const WORDS_PER_QUESTION = 25;
/* Has to match QUIZ_MAX_QUESTIONS in the backend. It capped a quiz at 5 while
   this offered 20, so asking for 20 silently gave you 5. */
const QUESTION_COUNTS = [3, 5, 10];

function countWords(text) {
  return (text || '').trim().split(/\s+/).filter(Boolean).length;
}
/* The largest count this text is long enough for. Rounded to the counts actually
   on offer, rather than to multiples of five, which stopped matching them. */
function questionsSupported(words) {
  const room = Math.floor(words / WORDS_PER_QUESTION);
  const fits = QUESTION_COUNTS.filter(n => n <= room);
  return fits.length ? fits[fits.length - 1] : QUESTION_COUNTS[0];
}

/* What the student picked on the "Make my quiz" screen */
const quizPrefs = { n: 5, kinds: 'mixed' };

/* Counts the seconds on the quiz loading screen, so a long wait does not look frozen */
let quizWaitTimer = null;
function tickQuizWait() {
  clearInterval(quizWaitTimer);
  const paint = () => {
    const el = $('#quizWaited');
    if (!el || quizCtx.status !== 'loading') { clearInterval(quizWaitTimer); return; }
    const s = Math.round((Date.now() - (quizCtx.startedAt || Date.now())) / 1000);
    // Past a minute, "Waiting 83 seconds so far" is harder to read than 1:23.
    const clock = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
    el.textContent = s < 5 ? '' : `Waiting ${clock} so far.`;
  };
  paint();
  quizWaitTimer = setInterval(paint, 1000);
}

async function loadQuizQuestions() {
  quizCtx.status = 'loading'; quizCtx.error = ''; quizCtx.answers = {};
  quizCtx.startedAt = Date.now();
  const token = ++quizRequestToken;
  renderQuizWorkspace();
  const supported = questionsSupported(countWords(quizCtx.text));
  const d = await callBackend('/quiz', {
    text: quizCtx.text,
    goal: null,
    n: Math.min(quizPrefs.n, supported),
    kinds: quizPrefs.kinds,
    // Follows the coach-language setting they already chose in Settings rather
    language: state.responseLanguage === 'auto' ? null : state.responseLanguage
  });
  // They backed out or picked another reading while this was in flight.
  if (token !== quizRequestToken) return;
  if (d && Array.isArray(d.questions) && d.questions.length) {
    quizCtx.questions = d.questions; quizCtx.source = 'ai'; quizCtx.status = 'ready';
  } else if (quizCtx.demoId === 'demo-lamp') {
    // The hand-written offline fallback quiz only covers this one story.
    quizCtx.questions = OFFLINE_QUIZ; quizCtx.source = 'demo'; quizCtx.status = 'ready';
  } else {
    quizCtx.status = 'error';
    quizCtx.error = CONFIG.API_BASE
      ? 'The question writer did not answer this time. Press Try again. If it keeps failing, tell your teacher the quiz service is down.'
      : 'Making a quiz needs the reading partner service, and it is not set up on this copy of the site. Tell your teacher.';
  }
  clearInterval(quizWaitTimer);
  renderQuizWorkspace();
}

function renderQuizWorkspace() {
  const el = $('#quizzesPane');
  const back = `<button class="ghost" id="quizBack" type="button">&larr; All readings</button>`;

  if (quizCtx.status === 'loading') {
    const n = Math.min(quizPrefs.n, questionsSupported(countWords(quizCtx.text)));
    el.innerHTML = `
      ${back}<h1>${esc(quizCtx.title)}</h1>
      <div class="card">
        <b>Writing your questions</b>
        <div class="quiz-wait" aria-hidden="true"><span></span></div>
        <p class="small" style="margin:10px 0 0">${n} question${n === 1 ? '' : 's'} from this text will appear on this screen when they are ready. On a slower computer this takes a minute or two.</p>
        <p class="small" id="quizWaited" style="margin:4px 0 0" role="status"></p>
      </div>`;
    wireQuizBack(el);
    tickQuizWait();
    return;
  }
  if (quizCtx.status !== 'ready') {
    const words = countWords(quizCtx.text);
    const supported = questionsSupported(words);
    if (quizPrefs.n > supported) quizPrefs.n = supported;
    el.innerHTML = `
      ${back}<h1>${esc(quizCtx.title)}</h1>
      <p class="hint">Make a short quiz based on this exact text.</p>
      ${quizCtx.status === 'error' ? `<div class="card"><b>Couldn't make the quiz</b><p style="margin:6px 0 0">${esc(quizCtx.error)}</p></div>` : ''}

      <label class="f" for="myQuizN">How many questions</label>
      <select id="myQuizN">${QUESTION_COUNTS.map(n => `<option value="${n}"${
        n === quizPrefs.n ? ' selected' : ''}${n > supported ? ' disabled' : ''}>${n} questions${
        n > supported ? ' (text is too short)' : ''}</option>`).join('')}</select>

      <label class="f" for="myQuizKinds">What kind of questions</label>
      <select id="myQuizKinds">
        <option value="mixed"${quizPrefs.kinds === 'mixed' ? ' selected' : ''}>A mix of both</option>
        <option value="literal"${quizPrefs.kinds === 'literal' ? ' selected' : ''}>Answers stated in the text</option>
        <option value="inferential"${quizPrefs.kinds === 'inferential' ? ' selected' : ''}>Answers you have to work out</option>
      </select>
      <p class="small" style="margin:6px 0 0">This text is ${words} words, so it can carry up to ${supported} question${supported === 1 ? '' : 's'}. The questions and answer choices come out in the language of the reading, and the explanations follow your language setting.</p>

      <div class="row" style="margin-top:14px"><button class="btn" id="makeQuiz" type="button">${quizCtx.status === 'error' ? 'Try again' : 'Make my quiz'}</button></div>`;
    wireQuizBack(el);
    const nSel = $('#myQuizN', el), kSel = $('#myQuizKinds', el);
    nSel.addEventListener('change', () => { quizPrefs.n = +nSel.value || 5; });
    kSel.addEventListener('change', () => { quizPrefs.kinds = kSel.value; });
    $('#makeQuiz', el).addEventListener('click', loadQuizQuestions);
    return;
  }

  const qs = quizCtx.questions;
  const done = Object.keys(quizCtx.answers).length;
  const score = Object.entries(quizCtx.answers).filter(([i, a]) => a === qs[i].answer).length;
  el.innerHTML = `
    ${back}<h1>${esc(quizCtx.title)}</h1>
    <p class="hint">${done ? `${score} of ${done} correct so far.` : 'Pick the best answer for each question.'}${quizCtx.source === 'demo' ? ' (Offline demo quiz)' : ''}${quizCtx.quizId ? ' Assigned by your teacher.' : ''}</p>
    ${quizCtx.instructions ? `<p class="small" style="margin:0 0 10px">${esc(quizCtx.instructions)}</p>` : ''}
    ${qs.map((q, i) => {
      const a = quizCtx.answers[i]; const answered = a !== undefined;
      return `<div class="card">
        <span class="tag">${esc(q.kind || 'Question')}</span>
        <div><b>${esc(q.q)}</b></div>
        ${q.options.map((o, j) => `<button class="opt ${answered ? (j === q.answer ? 'right' : j === a ? 'wrong' : '') : ''}" data-q="${i}" data-o="${j}" ${answered ? 'disabled' : ''}>${esc(o)}</button>`).join('')}
        ${answered ? `<div class="explain">${esc(q.why || '')}${q.evidence ? ` Look at: “${esc(q.evidence)}”` : ''}</div>` : ''}
      </div>`;
    }).join('')}
    ${quizCtx.quizId ? '' : '<div class="row"><button class="btn alt" id="newQuiz" type="button">Make new questions</button></div>'}`;
  wireQuizBack(el);
  $$('.opt', el).forEach(b => b.addEventListener('click', () => {
    const i = +b.dataset.q, j = +b.dataset.o;
    quizCtx.answers[i] = j;
    const correct = j === qs[i].answer;
    saveQuizAnswer(qs[i].q, qs[i].options[j], correct, qs[i].id || null);
    renderQuizWorkspace();
  }));
  const again = $('#newQuiz', el);
  if (again) again.addEventListener('click', loadQuizQuestions);
}
function wireQuizBack(el) {
  $('#quizBack', el).addEventListener('click', () => {
    setQuizCtx({ readingId: null, isDemo: false, demoId: null, isPasted: false, title: '', text: '', sessionId: null, questions: [], answers: {}, status: 'idle', error: '', source: null, quizId: null, instructions: null });
    renderQuizzesView();
  });
}
