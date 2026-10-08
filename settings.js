/* SETTINGS.JS -- the "Account" page shared by index.html and teacher.html.
   Fills whatever element you give it with: name, email, account type, a
   change-password form, and a light/dark switch.

   Also defines showToast(), a small popup used to confirm something just
   happened (saved, failed) -- shared so any page can use it, not just
   Settings.

   Needs `supa` (from auth.js) and a logged-in `profile` ({ id, role, full_name }). */
function escSettings(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function roleLabel(role) {
  return role ? role.charAt(0).toUpperCase() + role.slice(1) : '';
}

/* ---- Toast popups ----
   Slides in, sits a few seconds, fades out on its own. kind is 'ok'
   (green) or 'bug' (red) -- for things the system just did, not
   routine status text. */
function showToast(message, kind = 'ok') {
  let host = document.getElementById('toastHost');
  if (!host) {
    host = document.createElement('div');
    host.id = 'toastHost';
    document.body.appendChild(host);
  }
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  host.appendChild(el);
  setTimeout(() => {
    el.classList.add('leaving');
    setTimeout(() => el.remove(), 200);
  }, 3200);
}

async function renderSettings(containerId, profile) {
  const el = document.getElementById(containerId);
  if (!el || !profile) return;
  el.innerHTML = `<h1>Settings</h1><p class="hint">Loading your account…</p>`;

  const { data: { user } } = await supa.auth.getUser();
  const email = user ? user.email : '';

  el.innerHTML = `
    <h1>Settings</h1>
    <p class="hint">Manage your account and how Reading Partner looks on this device.</p>

    <div class="card">
      <h3 style="margin:0 0 10px">Your account</h3>
      <label class="f" for="setName">Name</label>
      <input type="text" id="setName" value="${escSettings(profile.full_name || '')}">
      <label class="f">Email <span class="small">(can't be changed here)</span></label>
      <input type="text" value="${escSettings(email)}" disabled>
      <label class="f">Account type <span class="small">(set by an admin)</span></label>
      <input type="text" value="${escSettings(roleLabel(profile.role))}" disabled>
      <div class="row" style="margin-top:14px">
        <button class="btn" id="saveNameBtn" type="button">Save name</button>
      </div>
    </div>

    <div class="card">
      <h3 style="margin:0 0 10px">Change password</h3>
      <label class="f" for="newPw">New password</label>
      <input type="password" id="newPw" autocomplete="new-password" minlength="6" placeholder="At least 6 characters">
      <label class="f" for="newPw2">Confirm new password</label>
      <input type="password" id="newPw2" autocomplete="new-password" minlength="6">
      <div class="row" style="margin-top:14px">
        <button class="btn" id="savePwBtn" type="button">Update password</button>
      </div>
    </div>

    <div class="card">
      <h3 style="margin:0 0 10px">Appearance</h3>
      <p class="small" style="margin:0 0 12px">Light, dark, or match this device's setting.</p>
      <div class="row">
        <button class="btn alt" id="themeLightBtn" type="button">Light</button>
        <button class="btn alt" id="themeDarkBtn" type="button">Dark</button>
        <button class="btn alt" id="themeAutoBtn" type="button">Match device</button>
      </div>
    </div>`;

  // --- Name ---
  document.getElementById('saveNameBtn').addEventListener('click', async () => {
    const btn = document.getElementById('saveNameBtn');
    const name = document.getElementById('setName').value.trim();
    if (!name) { showToast("Name can't be empty.", 'bug'); return; }
    btn.disabled = true;
    const { error } = await supa.from('profiles').update({ full_name: name }).eq('id', profile.id);
    btn.disabled = false;
    if (error) { showToast('Could not save: ' + error.message, 'bug'); return; }
    profile.full_name = name;
    const nameEl = document.getElementById('userName');
    if (nameEl) nameEl.textContent = name;
    showToast('Name updated.', 'ok');
  });

  // --- Password ---
  document.getElementById('savePwBtn').addEventListener('click', async () => {
    const btn = document.getElementById('savePwBtn');
    const p1 = document.getElementById('newPw').value;
    const p2 = document.getElementById('newPw2').value;
    if (p1.length < 6) { showToast('Use at least 6 characters.', 'bug'); return; }
    if (p1 !== p2) { showToast("Passwords don't match.", 'bug'); return; }
    btn.disabled = true;
    const { error } = await supa.auth.updateUser({ password: p1 });
    btn.disabled = false;
    if (error) { showToast('Could not update: ' + error.message, 'bug'); return; }
    document.getElementById('newPw').value = '';
    document.getElementById('newPw2').value = '';
    showToast('Password updated.', 'ok');
  });

  // --- Appearance ---
  document.getElementById('themeLightBtn').addEventListener('click', () => { setAppTheme('light'); showToast('Switched to light.', 'ok'); });
  document.getElementById('themeDarkBtn').addEventListener('click', () => { setAppTheme('dark'); showToast('Switched to dark.', 'ok'); });
  document.getElementById('themeAutoBtn').addEventListener('click', () => { setAppTheme(null); showToast('Now matching your device.', 'ok'); });
}

function setAppTheme(mode) {
  if (mode) document.documentElement.dataset.theme = mode;
  else delete document.documentElement.dataset.theme;
  try {
    if (mode) localStorage.setItem('theme', mode);
    else localStorage.removeItem('theme');
  } catch (e) { /* private browsing etc -- the switch still works for this page load */ }
}
// Reapply the saved theme immediately on load, so a refresh doesn't
// flash back to "match device" before Settings even renders.
(function applySavedTheme() {
  try {
    const saved = localStorage.getItem('theme');
    if (saved) document.documentElement.dataset.theme = saved;
  } catch (e) { /* private browsing etc -- falls back to match-device */ }
})();
