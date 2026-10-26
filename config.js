/* =====================================================================
   SUPABASE CONFIG
   Go to your Supabase project, then Settings, then API.
   Copy "Project URL" and the "anon public" key and paste them below.

   This key is SAFE to be public (it goes in index.html/GitHub). It only
   lets people do what your Row Level Security rules allow. The SECRET
   "service_role" key is different and must never go in this file.
   ===================================================================== */
const SUPABASE_URL = "https://kzkczgjawiyiinamxmdy.supabase.co";
const SUPABASE_ANON_KEY = "sb_publishable_eMH6cnljo2YhJaH2hrEzdA_qEQo0S2O";

/* =====================================================================
   BACKEND ADDRESS (read by app.js and teacher.html as APP_CONFIG.API_BASE)
   Local testing: "http://localhost:8000".
   Hosted site: the backend PC's tunnel address, e.g.
   "https://something.trycloudflare.com" (https, no port, no trailing slash).
   ===================================================================== */
window.APP_CONFIG = {
  // Changes every time cloudflared restarts: copy the https://....trycloudflare.com
  // address it prints (not the command), then commit and push to main.
  API_BASE: "http://localhost:8000",
};