/* AUTH.JS -- signup, login, and figuring out if someone's a student,
   teacher, or admin. Shared by index.html and teacher.html.

   To use on a page: have a <div id="authOverlay"> and an id="appRoot"
   for the protected content, load config.js + the Supabase script +
   this file, then call initAuth({ allow: ['student'], onReady: user => {...} }).
   allow = which roles may use this page; others get redirected. */

const supa = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

function escapeAuthHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function authHtml() {
  return `
  <div class="auth-card">
    <h1>Reading Partner</h1>
    <p class="auth-sub">Sign in to start reading, or create a new account.</p>
    <div class="auth-tabs">
      <button type="button" class="auth-tab" data-m="login" aria-selected="true">Log in</button>
      <button type="button" class="auth-tab" data-m="signup" aria-selected="false">Sign up</button>
    </div>
    <p class="auth-err" id="authError"></p>

    <form id="loginForm" class="auth-form">
      <label class="f" for="loginEmail">Email</label>
      <input type="email" id="loginEmail" autocomplete="email" required>
      <label class="f" for="loginPass">Password</label>
      <input type="password" id="loginPass" autocomplete="current-password" required minlength="6">
      <button class="btn auth-submit" type="submit" id="loginSubmit">Log in</button>
    </form>

    <form id="signupForm" class="auth-form" hidden>
      <label class="f" for="signupName">Your name</label>
      <input type="text" id="signupName" autocomplete="name" required>
      <label class="f">I am a</label>
      <div class="scale" id="roleChoice">
        <button type="button" data-role="student" aria-pressed="true">Student</button>
        <button type="button" data-role="teacher" aria-pressed="false">Teacher</button>
      </div>
      <label class="f" for="signupEmail">Email</label>
      <input type="email" id="signupEmail" autocomplete="email" required>
      <label class="f" for="signupPass">Password</label>
      <input type="password" id="signupPass" autocomplete="new-password" required minlength="6">
      <button class="btn auth-submit" type="submit" id="signupSubmit">Create account</button>
    </form>
  </div>`;
}

async function initAuth({ allow, onReady }) {
  const overlay = document.getElementById('authOverlay');
  const root = document.getElementById('appRoot');
  overlay.innerHTML = authHtml();

  let role = 'student';
  const err = document.getElementById('authError');
  const loginForm = document.getElementById('loginForm');
  const signupForm = document.getElementById('signupForm');

  // Each tab has its own form, so switching never carries typed text over.
  document.querySelectorAll('.auth-tab').forEach(t => t.addEventListener('click', () => {
    const mode = t.dataset.m;
    document.querySelectorAll('.auth-tab').forEach(x => x.setAttribute('aria-selected', String(x === t)));
    loginForm.hidden = mode !== 'login';
    signupForm.hidden = mode !== 'signup';
    err.textContent = '';
  }));
  document.querySelectorAll('#roleChoice button').forEach(b => b.addEventListener('click', () => {
    role = b.dataset.role;
    document.querySelectorAll('#roleChoice button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  }));

  loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    err.textContent = '';
    const btn = document.getElementById('loginSubmit');
    btn.disabled = true;
    try {
      const email = document.getElementById('loginEmail').value.trim();
      const password = document.getElementById('loginPass').value;
      const { data, error } = await supa.auth.signInWithPassword({ email, password });
      if (error) throw error;
      await afterLogin(data.user.id);
    } catch (ex) {
      err.textContent = ex.message || 'Something went wrong. Please try again.';
    } finally {
      btn.disabled = false;
    }
  });

  signupForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    err.textContent = '';
    const btn = document.getElementById('signupSubmit');
    btn.disabled = true;
    try {
      const name = document.getElementById('signupName').value.trim();
      const email = document.getElementById('signupEmail').value.trim();
      const password = document.getElementById('signupPass').value;
      const { data, error } = await supa.auth.signUp({ email, password });
      if (error) throw error;
      if (!data.session) { err.textContent = 'Check your email to confirm your account, then log in.'; return; }
      // give the new account a matching row in "profiles"
      const { error: pErr } = await supa.from('profiles').insert({ id: data.user.id, role, full_name: name });
      if (pErr) throw pErr;
      showSignupSuccess(name, role, () => afterLogin(data.user.id));
    } catch (ex) {
      err.textContent = ex.message || 'Something went wrong. Please try again.';
    } finally {
      btn.disabled = false;
    }
  });

  function showSignupSuccess(name, chosenRole, onContinue) {
    const card = overlay.querySelector('.auth-card');
    card.innerHTML = `
      <h1>You're all set</h1>
      <p class="auth-sub">Account created for <b>${escapeAuthHtml(name)}</b> as a <b>${escapeAuthHtml(chosenRole)}</b>.</p>
      <button class="btn auth-submit" type="button" id="continueBtn">Continue</button>`;
    document.getElementById('continueBtn').addEventListener('click', onContinue);
  }

  async function afterLogin(userId) {
    let { data: profile, error } = await supa.from('profiles').select('*').eq('id', userId).single();

    // An interrupted signup can leave a login with no "profiles" row --
    // rather than a dead end, make one now (defaulting to student).
    if (error && error.code === 'PGRST116') {
      const { data: { user } } = await supa.auth.getUser();
      const fallbackName = (user && user.email) ? user.email.split('@')[0] : 'User';
      const created = await supa.from('profiles')
        .insert({ id: userId, role: 'student', full_name: fallbackName })
        .select('*').single();
      profile = created.data; error = created.error;
      if (!error) err.textContent = 'Your account needed a quick repair. You are set up as a student; ask an admin to change your role if needed.';
    }

    if (error || !profile) { err.textContent = "Couldn't load your account (" + (error ? error.message : 'unknown error') + "). Try logging in again."; return; }
    if (!allow.includes(profile.role)) {
      err.textContent = `This page is for ${allow.join(' or ')}s. Your account is a ${profile.role} account.`;
      redirectForRole(profile.role);
      return;
    }
    overlay.hidden = true;
    root.hidden = false;
    onReady(profile);
  }

  function redirectForRole(r) {
    setTimeout(() => {
      if (r === 'teacher' || r === 'admin') location.href = 'teacher.html';
      else location.href = 'index.html';
    }, 1200);
  }

  // already logged in? skip straight to the app
  const { data: { session } } = await supa.auth.getSession();
  if (session) await afterLogin(session.user.id);
}

async function signOut() {
  await supa.auth.signOut();
  location.reload();
}
