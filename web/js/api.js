/* api.js — cliente HTTP mínimo para el backend stdlib */
const API = {
  async get(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error('GET ' + path + ' -> ' + r.status);
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!r.ok) throw new Error('POST ' + path + ' -> ' + r.status);
    return r.json();
  }
};
